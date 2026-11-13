"""Restore Stage2 inference directly from weights; sidecar JSON is optional."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from safetensors import safe_open
import torch

from tcp4d.droid_checkpoint import weight_file
from tcp4d.inference import _remove_training_prefix

# Attention head counts and sampling schedules are not encoded in tensor shapes.
# These are the published TCP4D Stage2 defaults; callers can override them.
DEFAULTS = {
    'action_heads': 32, 'action_dropout': .2, 'action_gradient_checkpointing': False,
    'global_token_grid': [14, 19], 'history_frames': 4, 'prediction_horizon': 32,
    'text_max_length': 128, 'sampling_steps': 4, 't5_model': 'google-t5/t5-base',
    'decode_geometry': True,
}


def canonical_key(key):
    key = _remove_training_prefix(key)
    return 'tcp4d.' + key[4:] if key.startswith('arc.') else key


def read_optional_config(directory):
    path = directory/'config.json'
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    except (OSError, ValueError) as error:
        print(f'Ignoring optional {path}: {error}', flush=True)
        return {}


def inspect_stage2_checkpoint(checkpoint, overrides=None):
    path = weight_file(checkpoint)
    if path.suffix != '.safetensors':
        raise ValueError('Stage2 inference expects a model.safetensors file or its directory')
    metadata = read_optional_config(path.parent)
    config = {key: metadata.get(key, value) for key,value in DEFAULTS.items()}
    if overrides:
        config.update({key:value for key,value in overrides.items() if value is not None})
    with safe_open(path,framework='pt',device='cpu') as archive:
        mapping = {}
        for raw in archive.keys():
            key = canonical_key(raw)
            if key in mapping:
                raise ValueError(f'Duplicate checkpoint tensor: {key}')
            mapping[key] = raw
        def shape(key):
            if key not in mapping:
                raise ValueError(f'Stage2 checkpoint is missing required tensor: {key}')
            return tuple(archive.get_slice(mapping[key]).get_shape())
        for kind,width in [('action',20),('state',14)]:
            for suffix in ['offset','scale']:
                key = f'{kind}_{suffix}'
                if shape(key) != (1,width):
                    raise ValueError(f'Expected {key} shape [1,{width}]')
                value = archive.get_tensor(mapping[key])
                if not torch.isfinite(value).all() or (suffix == 'scale' and torch.any(value <= 0)):
                    raise ValueError(f'{key} must be finite, with positive normalization scales')
        if shape('tcp4d.tcp_visual_query_encoder.arm_embedding')[0] != 2:
            raise ValueError('The Stage2 policy needs a checkpoint with left/right TCP embeddings')
        offsets = shape('tcp4d.tcp_visual_query_encoder.offset_embedding')[0]
        query_window = math.isqrt(offsets)
        blocks = sorted({int(match[1]) for key in mapping
                         if (match := re.match(r'dit\.blocks\.(\d+)\.',key))})
        if blocks != list(range(len(blocks))) or not blocks:
            raise ValueError('Stage2 attention blocks are missing or non-contiguous')
        config.update(action_dim=shape('action_encoder.projection.weight')[0], action_depth=len(blocks),
                      condition_dim=shape('global_encoder.projection.1.weight')[0],
                      action_output_dim=shape('dit.output_projection.weight')[0],
                      tcp_query_window_size=query_window)
        if query_window*query_window != offsets:
            raise ValueError('TCP neighborhood embeddings do not form a square window')
        prefix = 'wrist_encoder.encoder.'
        patch = shape(prefix+'embeddings.patch_embeddings.projection.weight')
        layers = {int(match[1]) for key in mapping
                  if (match := re.match(r'wrist_encoder\.encoder\.encoder\.layer\.(\d+)\.',key))}
        tokens = shape(prefix+'embeddings.position_embeddings')[1]-1
        grid = math.isqrt(tokens)
        if grid*grid != tokens or not layers:
            raise ValueError('Cannot recover the DINOv2 wrist encoder dimensions')
        width,channels,patch_size,_ = patch
        wrist = dict(metadata.get('wrist_encoder_config') or {})
        wrist.update(hidden_size=width,num_hidden_layers=max(layers)+1,
                     num_attention_heads=wrist.get('num_attention_heads',width//64),
                     mlp_ratio=shape(prefix+'encoder.layer.0.mlp.fc1.weight')[0]/width,
                     image_size=grid*patch_size,patch_size=patch_size,num_channels=channels,
                     qkv_bias=prefix+'encoder.layer.0.attention.attention.query.bias' in mapping,
                     use_swiglu_ffn=False)
        config['wrist_encoder_config'] = wrist
        if not 1 <= config['prediction_horizon'] <= shape('action_encoder.position.weight')[0]:
            raise ValueError('Prediction horizon exceeds the checkpoint action-position capacity')
    return path, config


def restore_stage2_weights(policy, path):
    """Strictly restore all parameters, permitting omitted copies of tied weights."""
    from safetensors.torch import load_file, load_model
    with safe_open(path,framework='pt',device='cpu') as archive:
        original_keys = list(archive.keys())
    if all(canonical_key(key) == key for key in original_keys):
        load_model(policy,str(path),strict=True,device='cpu')
        return
    state = {canonical_key(key):value for key,value in load_file(str(path),device='cpu').items()}
    aliases = {}
    for name,tensor in policy.state_dict().items():
        identity = (tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tuple(tensor.shape),tensor.stride())
        aliases.setdefault(identity,[]).append(name)
    for names in aliases.values():
        source = next((name for name in names if name in state),None)
        if source is not None:
            for name in names:
                state.setdefault(name,state[source])
    policy.load_state_dict(state,strict=True)

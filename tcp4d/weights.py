"""Check the arm count before allocating a policy or loading large weights."""
from pathlib import Path
import torch
from tcp4d.droid_checkpoint import weight_file

def require_two_arm_stage1(checkpoint):
    path = weight_file(checkpoint)
    key = "tcp_visual_query_encoder.arm_embedding"
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            shape = handle.get_slice(key).get_shape() if key in handle.keys() else None
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k.removeprefix("model.").removeprefix("module."): v for k,v in state.items()}
        shape = list(state[key].shape) if key in state else None
    if shape is None or shape[0] != 2:
        raise ValueError("RoboTwin Stage2 requires a trained two-arm Stage1 checkpoint; "
                         "single-arm DROID checkpoints cannot initialize this policy.")


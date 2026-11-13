# TCP4D Generator Deployment

We provide a WebSocket server template for deploying the **TCP4D Generator
(Stage2)** with the released
[TCP4D-stage2-RoboTwin checkpoint](https://huggingface.co/HarrisonPENG/TCP4D-stage2-RoboTwin).
It takes third-person RGB history, initial 2D TCP queries, current wrist RGB,
current robot state and a language instruction, and returns **32 future steps**
of dual-arm TCP poses and gripper states. The released checkpoint uses **4 history
frames** and **4 sampling steps**; history length and prediction horizon are read
from the checkpoint.

The server is stateless: the client supplies the complete observation window on
every request and updates the first-frame TCP queries whenever that window moves.
Robot observation capture, camera-to-base transforms, inverse kinematics and action
execution belong in the client integration.

## Setup and launch

Follow the installation instructions in the [repository README](../README.md)
first. Use the same `tcp4d` environment: Python 3.11, PyTorch 2.9.1 + CUDA 12.8,
torchvision 0.24.1, Transformers 4.55.4 and Accelerate 1.10.1. Download the Stage2
checkpoint to `checkpoints/TCP4D-stage2-RoboTwin/`.

Run all commands from the repository root:

```bash
conda activate tcp4d
python -m pip install -r deploy/requirements.txt
python -m deploy.server \
  --checkpoint checkpoints/TCP4D-stage2-RoboTwin \
  --device cuda:0 --host 0.0.0.0 --port 8000
```

`python deploy/server.py` is also supported. The default checkpoint is
`checkpoints/TCP4D-stage2-RoboTwin`. `--checkpoint` accepts either a directory
containing the checkpoint files or an experiment directory containing
`final_checkpoint/`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | `checkpoints/TCP4D-stage2-RoboTwin` | Stage2 checkpoint directory |
| `--device` | `auto` | CUDA when available, otherwise CPU; also accepts `cuda:0` or `cpu` |
| `--dtype` | `auto` | BF16/FP16 on CUDA, FP32 on CPU; explicit choices are `float32`, `float16`, `bfloat16` |
| `--sampling-steps` | Checkpoint value (4 for the released model) | Action sampling steps |
| `--seed` | `42` | Sampling seed, reset for each request |
| `--host` / `--port` | `0.0.0.0` / `8000` | WebSocket bind address |
| `--t5-model` | Checkpoint configuration, with resource fallback | Optional local T5-base directory or model identifier |

The server loads the model once and starts listening only after it is ready. A
complete checkpoint must include `config.json`, `model.safetensors`,
`action_norm.json` and `state_norm.json`; normalization statistics are checked
against the saved model buffers. The wrist DINOv2 configuration and weights are
restored from the checkpoint. T5 resources are loaded separately: if the saved
absolute T5 path is unavailable, the loader checks for `t5-base/` inside or beside
the checkpoint, then falls back to `google-t5/t5-base` through Hugging Face. For
offline deployment, provide an available local directory:

```bash
python -m deploy.server \
  --checkpoint checkpoints/TCP4D-stage2-RoboTwin \
  --device cuda:0 --t5-model /path/to/local/t5-base
```

One process loads one policy and serializes inference across connections. Inference
runs in a worker thread, allowing other connections to send health checks while
it is running. Shutdown waits for an active inference to finish. Each request
uses an independently seeded sampling generator; results do not depend on earlier
requests' random state.

## Observations and preprocessing

All image sizes below are **width × height**. Send original RGB images; the server
applies the preprocessing used during training.

| Input | Client representation | Server / model preprocessing |
| --- | --- | --- |
| Third-person history | 320×240 RGB, `uint8 [0,255]` | Reflect-pad 1 pixel left/right and 6 top/bottom to 322×252; normalize to `[-1,1]` |
| Current left/right wrist images | 320×240 RGB, `uint8 [0,255]` | Convert to `[0,1]`; the wrist encoder uses bicubic resize to 256×192, reflect-padding to 266×196 and ImageNet normalization |
| Camera intrinsics and TCP queries | Pixel coordinates in the original, unpadded 320×240 image | Shift the principal point and queries by `(1,6)` |
| Current robot state | 12 joint angles in radians and two grippers in `[0,1]` | Apply the checkpoint's state normalization inside the model |

Images must be RGB **HWC NumPy arrays**. Convert OpenCV BGR images to RGB on the
client. Encoded JPEG bytes, floating-point images and other resolutions are not
accepted. Do not pre-pad, pre-normalize or resize wrist images to 256×192.

Supply 1–4 third-person frames, oldest first. Short histories repeat their earliest
frame: `[0] → [0,0,0,0]`, `[0,1] → [0,0,0,1]`, `[0,1,2] → [0,0,1,2]`.
More than 4 frames are rejected for this checkpoint. Wrist images and current
robot state must correspond to the latest history frame. TCP queries must locate
the left and right TCP in the **first supplied frame**.

The API uses frame indices and does not accept timestamps. Match the observation
cadence to the data used for your checkpoint. The bundled RoboTwin sample records
`frequency_hz=15` in `metadata.json`; its next 32 steps refer to subsequent sampled
frames. This does not imply a 15 Hz server inference rate. Replanning and how many
predicted steps to execute are client decisions; the offline inference script's
20-step window stride is not applied by this server.

## WebSocket API (protocol version 1)

Connect to `ws://SERVER_IP:8000`. The first server message has `type="metadata"`
and describes the protocol version, history length, prediction horizon, image
sizes, coordinate frames, arm order and gripper convention. Read and validate it
before sending requests.

Encode and decode every message with `deploy.protocol.pack_message` and
`unpack_message`, and send **binary WebSocket messages**. Clients need only
`deploy/protocol.py`, NumPy, MessagePack and websockets; model dependencies and
PyTorch are not required. This is a custom protocol, distinct from OpenPI and
`msgpack-numpy`.

The message limit is **16 MiB**. Use `max_size=MAX_MESSAGE_BYTES` and
`compression=None` on the client; a full observation exceeds websockets' default
1 MiB limit. Responses follow request order within a connection. Wait for a
prediction before sending the next prediction request.

### Prediction request

| Field | Type / shape | Meaning |
| --- | --- | --- |
| `type` | `"predict"` | Request type |
| `request_id` | String or integer | Returned unchanged in the response |
| `images` | `uint8 [T,240,320,3]` | Third-person history, `1≤T≤4`, oldest first |
| `tcp_query_points` | Numeric `[2,2]` | Left/right `(u,v)` in the first unpadded frame; `0≤u<320`, `0≤v<240` |
| `intrinsics` | Numeric `[3,3]` | OpenCV camera matrix K for the 320×240 view; shared across the history window |
| `wrist_images` | `uint8 [2,240,320,3]` | Current left/right wrist RGB |
| `current_robot_state` | Numeric `[14]` | Left six joints, left gripper, right six joints, right gripper |
| `instruction` | Non-empty string | Task instruction |

Small numeric fields may be arrays or lists; images must be `uint8` NumPy arrays.
All numeric inputs must be finite. K must be nonsingular, have positive `fx,fy`
and have last row `[0,0,1]`.

Grippers occupy state indices **6 and 13**: **0=closed, 1=open**. The bundled
RoboTwin `robot_state.npy` already uses `[0,1]`; send it directly. For another
robot, map measured gripper openings into this convention on the client according
to its calibration. The server does not infer units or apply the old task-specific
0.10 m conversion. Do not normalize the joint angles yourself.

No ground-truth history TCP poses, future images, depth or camera extrinsics are
required by the server.

### Successful response

Common fields are `type="prediction"`, the original `request_id`, `success=True`
and `inference_ms`. Timing includes preprocessing, model execution and transfer
back to CPU, and excludes queuing and network transmission.

Shapes below use `H=32` for the released checkpoint and `T` for the number of real
input frames, excluding repeated padding frames. Every arm dimension is ordered
**left, right**.

| Field | Type / shape | Meaning |
| --- | --- | --- |
| `action_position` | `float32 [H,2,3]` | Absolute future TCP positions in meters |
| `action_rotation` | `float32 [H,2,3,3]` | Future TCP rotation matrices |
| `action_gripper` | `int64 [H,2]` | `1=open`, `0=closed` |
| `action_gripper_score` | `float32 [H,2]` | Raw score; `>=0` means open, not a probability |
| `future_step_indices` | `int64 [H]` | Steps 1 through H |
| `history_position` | `float32 [T,2,3]` | Recovered TCP positions for real history frames |
| `history_rotation` | `float32 [T,2,3,3]` | Recovered history TCP rotations |
| `history_confidence` | `float32 [T,2]` | Raw TCP-head confidence, not a probability |
| `history_query_points_px` | `float32 [T,2,2]` | Recovered TCP projections in unpadded image pixels |
| `history_valid` | `bool [T,2]` | Validity mask for recovered history entries |
| `history_padding` | Integer | Number of repeated earliest frames, `4-T` |

Future poses use the **latest observation's OpenCV camera frame**: x right, y
down, z forward. Positions are absolute, not deltas. Rotation matrices transform
TCP-local vectors into that camera frame. Convert poses to the robot base frame
and solve IK in the client, using the same TCP definition as the training data.

History poses use each observation's own camera frame. Invalid history entries
have zero-filled positions, rotations, confidences and projections; always inspect
`history_valid`. Projections outside the image are not clipped to its border.

For a moving window `[f0,f1,f2,f3] → [f1,f2,f3,f4]`, use the previous result's
`history_query_points_px[1]` as the new first-frame queries only if both arms in
`history_valid[1]` are true. Otherwise, locate the TCPs again on the client. Use
recovered history projections for this update; future predicted actions do not
provide the observed first-frame queries. The server does not propagate queries.

### Health checks and errors

Send `{"type": "health", "request_id": "ping"}` with the same binary codec. The
response is `{"type": "health", "request_id": "ping", "success": True,
"ready": True}`. The server accepts connections only after model loading finishes.

Recoverable errors have this structure:

```python
{
    "type": "error", "request_id": "window-0", "success": False,
    "error": {"code": "invalid_request", "message": "..."},
}
```

Codes are `invalid_request`, `prediction_failed` and `inference_error`. Error
responses contain no `action_*` fields; check `success` before consuming actions.
The connection remains usable after these errors. Messages exceeding 16 MiB cause
the WebSocket layer to close the connection. If the request identifier cannot be
decoded, the error response uses `request_id=None`.

### Clients in other languages

The root message is a MessagePack map with string keys. NumPy arrays use this map:

```python
{"__ndarray__": True, "dtype": "<f4", "shape": [2, 2], "data": raw_bytes}
```

`dtype` follows NumPy `dtype.str`, e.g. `|u1`, `<f4`, `<i8`, `|b1`. `data` is a
binary payload in C order. Only bool, integer and real floating-point arrays are
supported; byte length must match shape and dtype. Object arrays and pickle are
not supported.

## Python client example

Run this example from the repository root after starting the server. It uses the
same RoboTwin sample as the root README and obtains initial queries by projecting
the sample's ground-truth TCP positions. A live robot client can initialize queries
with manual selection or calibrated TCP localization instead. The example needs
Pillow to read the sample PNG files; it is included in the project environment.

```python
import asyncio
import json
from pathlib import Path

import numpy as np
from PIL import Image
from websockets.asyncio.client import connect

from deploy.protocol import MAX_MESSAGE_BYTES, PROTOCOL_VERSION, pack_message, unpack_message

episode = Path("assets/samples/robotwin/adjust_bottle/episode_0000000")
frame_indices = [0, 1, 2, 3]
metadata = json.loads((episode / "metadata.json").read_text())

def rgb(view, frame):
    with Image.open(episode / "images" / view / f"{frame:06d}.png") as image:
        return np.asarray(image.convert("RGB")).copy()

K = np.load(episode / "intrinsics/third_views.npy").astype(np.float32)
xyz = np.stack([
    np.load(episode / "TCP_third" / f"{arm}_state.npy")[frame_indices[0], :3]
    for arm in ("left", "right")
])
pixels = xyz @ K.T
queries = pixels[:, :2] / pixels[:, 2:3]
state = np.load(episode / "robot_state.npy")[frame_indices[-1]].astype(np.float32)
request = {
    "type": "predict", "request_id": "window-0",
    "images": np.stack([rgb("third_views", i) for i in frame_indices]),
    "tcp_query_points": queries, "intrinsics": K,
    "wrist_images": np.stack([
        rgb(side, frame_indices[-1]) for side in ("left_view", "right_view")
    ]),
    "current_robot_state": state,
    "instruction": metadata["instructions"][0],
}

async def predict():
    async with connect("ws://127.0.0.1:8000", max_size=MAX_MESSAGE_BYTES,
                       compression=None, proxy=None) as ws:
        server_metadata = unpack_message(await ws.recv())
        assert server_metadata["type"] == "metadata"
        assert server_metadata["protocol_version"] == PROTOCOL_VERSION
        assert server_metadata["history_frames"] == 4
        assert server_metadata["prediction_horizon"] == 32
        await ws.send(pack_message({"type": "health", "request_id": "ping"}))
        health = unpack_message(await ws.recv())
        assert health["success"] and health["ready"]
        await ws.send(pack_message(request))
        result = unpack_message(await ws.recv())
        if not result["success"]:
            raise RuntimeError(result["error"])
        print(result["action_position"].shape)  # (32, 2, 3)
        print(result["action_rotation"].shape)  # (32, 2, 3, 3)
        print(result["action_gripper"].shape)   # (32, 2), 0/1
        print(result["history_valid"])

asyncio.run(predict())
```

This checks metadata, readiness and one inference request without executing any
robot actions. For another machine, change the WebSocket URL to the server's
address. A client running outside the model environment can install its transport
dependencies with `python -m pip install -r deploy/requirements.txt` and use a copy
of `deploy/protocol.py`; Pillow is needed only when reading images as in this
example.

# Complete pipeline examples

The two episodes in this directory are complete preprocessed exports copied from the working 4RC data collection. All original frames and arrays are retained. Machine-local `/home/...` and `/mnt/...` provenance strings in JSON were replaced by portable `upstream:<filename>` identifiers; labels, calibration and image pixels were not changed.

```text
datasets/
├── droid/AUTOLab__Fri_Jul_14_16:44:09_2023/
│   ├── metadata.json
│   ├── images/<camera>/000000.png
│   ├── depths/<camera>/000000.png
│   ├── intrinsic/<camera>.npy
│   ├── extrinsic/<camera>.npy
│   └── TCP/<camera>/{metadata.json,state.npy}
├── robotwin/adjust_bottle/episode_0000000/
│   ├── metadata.json
│   ├── images/{third_views,left_view,right_view,head_view}/000000.png
│   ├── depths/<view>/000000.png
│   ├── intrinsics/<view>.npy
│   ├── extrinsics/<view>.npy
│   ├── TCP_third/{metadata.json,left_state.npy,right_state.npy}
│   ├── TCP_head/{metadata.json,left_state.npy,right_state.npy}
│   ├── robot_state.npy
│   └── robot_action.npy
└── splits/droid/{train_set.txt,val_set.txt}
```

Check the actual filenames in each episode; additional upstream arrays and timestamps are preserved as supplied.

| Field | DROID | RoboTwin |
| --- | --- | --- |
| Frames / rate | 126 per camera / fixed 15 Hz | 150 / metadata `frequency_hz=15` |
| Camera streams | `22008760`, `24400334` | `third_views`, `left_view`, `right_view`, `head_view` |
| RGB | 320×180, RGB PNG | 320×240, RGB PNG |
| Depth | uint16 PNG, millimetres | uint16 PNG, millimetres |
| Intrinsics | `[3,3]`, per camera | `[3,3]`, per view |
| Extrinsics | Static `[4,4]` robot-base/world to camera | Per-frame `[T,3,4]` world to camera |
| TCP | `[T,7]`, one arm per camera | Left/right `[T,7]` in the selected camera |
| Gripper labels | Continuous openness `[0,1]` | Binary openness `0/1` |

TCP columns are `x,y,z,roll,pitch,yaw,gripper_open`: metres, radians, and `Rz(yaw) @ Ry(pitch) @ Rx(roll)`. OpenCV camera axes are +x right, +y down, +z forward. RoboTwin state columns are left six joints, left gripper, right six joints, right gripper; Stage2 reads real state rather than predicted TCP as joint input.

The examples are assigned entirely to the training side for pipeline checks. The DROID validation TXT contains only a comment; RoboTwin sample validation is disabled. Do not report these examples as held-out evaluation or mix them into a claimed evaluation split without auditing their source membership.

`manifest.json` records each episode's file count, byte count and SHA-256 tree digest. The digest is computed over sorted relative file paths, a NUL byte, the file's SHA-256 hex digest and a newline for every file. Generated cache/index files are excluded. The repository's CC BY 4.0 license for TCP4D code and model weights does not relicense these dataset assets; retain the source datasets' applicable terms and attribution. See [NOTICE](../../NOTICE).

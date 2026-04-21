# Teammate Setup Guide — End-to-End NER + GPSR

This document lists everything a new team member needs, in addition to GitHub
access, to reproduce the end-to-end BERT-NER + GPSR demo on a fresh machine.

---

## 1. Git — branches to check out

All the work lives on feature branches across six repositories. After cloning
the workspace, a teammate must switch each repo to the correct branch.

| Repo                                   | Branch                              | Purpose                                                                                                 |
| -------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `tue-robotics/ner_model`               | `feature/refined-ner-model`         | NER package: parser, mapper, data, README, this doc                                                     |
| `tue-robotics/hmi`                     | `feature/bert-gpsr`                 | Swaps `CFGParser` → `NERParser` in `common.py::parse_sentence`                                          |
| `tue-robotics/conversation_engine`     | `feature/parser-change`             | Bypasses CFG grammar validation; uses NER directly for commands                                         |
| `tue-robotics/tue_robocup`             | `bug/new_classification`            | Reads YOLO-assigned `entity.etype` from ED instead of calling the dead `ed.classify` service            |
| `tue-robotics/hero_bringup`            | `feature/remove-image-recognition`  | Disables the dead TensorFlow `object_recognition` launch include                                        |
| `tue-robotics/robot_launch_files`      | `master` (no changes)               | —                                                                                                       |

### One-liner to switch them all

Run from the `repos/github.com/tue-robotics/` directory:

```bash
for pair in \
  "ner_model:feature/refined-ner-model" \
  "hmi:feature/bert-gpsr" \
  "conversation_engine:feature/parser-change" \
  "tue_robocup:bug/new_classification" \
  "hero_bringup:feature/remove-image-recognition"
do
  repo="${pair%%:*}"; branch="${pair##*:}"
  (cd "$repo" && git fetch && git checkout "$branch" && git pull)
done
```

> **Note**: these branches are unmerged. Open PRs and merge once validated.

---

## 2. Binaries and model files NOT in GitHub

### 2a. NER model weights — must be obtained separately

| File         | Path                          | Size   | Tracked in git?                 | How to get it                                                                                                           |
| ------------ | ----------------------------- | ------ | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `model.pth`  | `ner_model/data/model.pth`    | 433 MB | **No** — ignored via `*.pth`    | Download from Hugging Face: [PanosHuggingFace/bert-ner/model.pth](https://huggingface.co/PanosHuggingFace/bert-ner/resolve/main/model.pth) |
| `vocab.slot` | `ner_model/data/vocab.slot`   | < 1 KB | **Yes**                         | Comes with the repo                                                                                                     |

Quick download (pick one):

```bash
cd /home/amigo/ros/noetic/repos/github.com/tue-robotics/ner_model/data

# with wget
wget -O model.pth https://huggingface.co/PanosHuggingFace/bert-ner/resolve/main/model.pth

# or with curl
curl -L -o model.pth https://huggingface.co/PanosHuggingFace/bert-ner/resolve/main/model.pth
```

Verify the download (expected ≈ 413 MiB / 433,429,702 bytes):

```bash
ls -lh data/model.pth
```

### 2b. YOLO + SAM ONNX weights — required for object classification

The `ed_sensor_integration` YOLO+SAM pipeline expects three ONNX files:

```
$(TUE_ENV_WS_DIR)/build/yolo_onnx_ros/resources/yolo11m/yolo11m.onnx
$(TUE_ENV_WS_DIR)/build/sam_onnx_ros/resources/speed_SAM/SAM_encoder.onnx
$(TUE_ENV_WS_DIR)/build/sam_onnx_ros/resources/speed_SAM/SAM_mask_decoder.onnx
```

These are normally fetched by `yolo_onnx_ros` / `sam_onnx_ros` CMake at build
time. If `tue-make` reports any of them missing, the teammate needs to acquire
them through the same mechanism you used. Paths are configured in
`hero_bringup/parameters/world_modeling/world_model_plugin_rgbd.yaml`.

### 2c. TensorFlow `image_recognition` files — NOT needed anymore

The old pipeline expected `~/MEGA/data/<ROBOT_ENV>/models/image_recognition_tensorflow/{output_graph.pb, output_labels.txt}`.
These are **no longer required**:

- `hero_bringup/feature/remove-image-recognition` disables the TF node launch.
- `tue_robocup/bug/new_classification` stopped calling `/ed/classify`.

---

## 3. Python dependencies

| Package        | Version       | Where it lives on the reference machine                                         |
| -------------- | ------------- | ------------------------------------------------------------------------------- |
| `torch`        | `2.3.1+cu121` | `~/.local/lib/python3.8/site-packages` (user-level)                             |
| `transformers` | `4.46.3`      | `/home/amigo/ros/noetic/.env/venv/lib/python3.8/site-packages` (ros-noetic venv)|

> `challenge_gpsr/gpsr.py` has the shebang `#!/usr/bin/python`, i.e. it uses
> the **system** interpreter, while most other ROS nodes run inside the
> `ros-noetic` venv. Both interpreters need `torch` and `transformers`:

```bash
# Into the ROS venv (preferred for most nodes)
pip install "torch==2.3.1" "transformers==4.46.3"

# Into system python (because gpsr.py has a system-python shebang)
/usr/bin/python3 -m pip install --user "torch==2.3.1" "transformers==4.46.3"
```

> **Longer-term**: add `tue-env` install targets so a clean box gets these via
> `tue-install-target` — not done yet.

---

## 4. System-level prerequisites

### 4a. CUDA runtime symlink (for `yolo_onnx_ros`)

`yolo_onnx_ros` links against `/usr/lib/x86_64-linux-gnu/libcudart.so`, which
the standard CUDA 12 install does not create. If `tue-make yolo_onnx_ros`
errors with *"No rule to make target `/usr/lib/x86_64-linux-gnu/libcudart.so`"*:

```bash
sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/x86_64-linux-gnu/libcudart.so
```

### 4b. Workspace symlink for `ner_model`

Catkin only discovers packages it can see under `system/src/`. Because
`ner_model` lives under `repos/github.com/tue-robotics/`, it must be symlinked:

```bash
ln -s /home/amigo/ros/noetic/repos/github.com/tue-robotics/ner_model \
      /home/amigo/ros/noetic/system/src/ner_model
```

This has to be done on every new machine — without it, `tue-make ner_model`
reports *"Given package 'ner_model' is not in the workspace"*.

---

## 5. Build

Once 1–4 are in place:

```bash
tue-make ner_model hmi conversation_engine robot_smach_states hero_bringup
# or simply:
tue-make
```

### Sanity check imports

```bash
# ROS venv
python3 -c "from ner_model.parser import NERParser; print('ok')"

# System python (used by gpsr.py)
/usr/bin/python3 -c "import torch, transformers; print('ok')"
```

---

## 6. Running end-to-end

Reference sequence (assumes `ROBOT_ENV=impuls`, `ROBOT=hero`):

```bash
# Terminal 1 — start the simulator (Gazebo + core robot stack)
hero-start

# Terminal 2 — free mode (navigation + world model + action server)
hero-free-mode

# Terminal 3 — (optional) RViz for visualisation
hero-rviz

# Terminal 4 — GPSR challenge, then immediately kill the random answerer
#              so it doesn't intercept HMI goals
rosrun challenge_gpsr gpsr.py _robot_name:=hero _test_mode:=true _skip:=true
rosnode kill /hero/hmi/random_answerer
```

Once GPSR prompts *"state your command"*, send a sentence directly on the
HMI string topic from any terminal:

```bash
rostopic pub /hero/hmi/string std_msgs/String \
    "data: 'get the coke from the dining table'" --once
```

---

## 7. Supporting docs

- `ner_model/README.md` — build and testing flow
- `ner_model/docs/extending_vocabulary.md` — adding new locations / objects / actions
- `tue_robocup/robocup_knowledge/src/robocup_knowledge/environments/impuls/common.py`
  — the `impuls` environment's known vocabulary (what the robot can resolve)

---

## 8. Open issues and rough edges

| Item                                                                                 | Impact                                                                                                   | Workaround                                                                              |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `model.pth` not tracked and no official distribution channel                         | Teammate has to get it manually from the maintainer                                                      | Set up Git LFS or a MEGA share; add a `tue-env` install target                          |
| No `tue-env` targets for `torch` / `transformers`                                    | Teammate must `pip install` manually in two interpreters                                                 | Add `install.yaml` targets under `tue-env-targets`                                      |
| Only `free_mode.launch` has `image_recognition.launch` disabled                      | If a teammate runs another launch file directly, the TF node still tries to start and dies (noisy logs) | Prefer `free_mode.launch`, or gut `image_recognition.launch` itself in `robot_launch_files` |
| Five feature branches are unmerged                                                   | Requires manual checkout of each repo                                                                    | Open PRs and merge once validated                                                       |
| `gpsr.py` shebang is `#!/usr/bin/python` (system interpreter)                        | Requires `torch` / `transformers` in **system** python as well as the venv                               | Leave as-is, or change the shebang to use the venv                                      |

---

## 9. Pre-flight checklist (copy-paste for teammates)

- [ ] Cloned all six `tue-robotics/*` repos
- [ ] Ran the branch-switch one-liner from section 1
- [ ] Downloaded `model.pth` from [Hugging Face](https://huggingface.co/PanosHuggingFace/bert-ner/resolve/main/model.pth) into `ner_model/data/`
- [ ] YOLO+SAM `.onnx` files are present under `$TUE_ENV_WS_DIR/build/{yolo,sam}_onnx_ros/resources/...`
- [ ] `pip install torch==2.3.1 transformers==4.46.3` in **both** the ros-noetic venv and system python
- [ ] `sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/x86_64-linux-gnu/libcudart.so` (if needed)
- [ ] `ln -s .../repos/.../ner_model /home/$USER/ros/noetic/system/src/ner_model`
- [ ] `tue-make` completes without errors
- [ ] Sanity imports from section 5 succeed
- [ ] `hero-start` + `hero-free-mode` (+ optional `hero-rviz`) bring up the stack cleanly
- [ ] `rosrun challenge_gpsr gpsr.py ...` reaches the "state your command" prompt
- [ ] `rostopic pub /hero/hmi/string ...` executes `get the coke from the dining table` end-to-end

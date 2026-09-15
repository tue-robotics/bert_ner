# ner_model

BERT-based NER model for slot filling, integrated as a catkin package to replace the CFG grammar parser in the HMI pipeline.

## Prerequisites

- ROS Noetic workspace with `hmi`, `action_server`, and `tue_robocup` packages
- Python dependencies: `torch`, `transformers`, `sentence-transformers`
- Model weights: `model.pth` (~433 MB). `vocab.slot` ships with the source; the
  weights do not and must be installed manually, see
  [docs/teammate_setup.md](docs/teammate_setup.md).

### Where the weights are looked up

Nothing is downloaded automatically. `model.pth` is searched for in this order,
so it can live outside the repository:

1. the `ner_model/model_path` ROS parameter, or the `model_path` argument of
   `load_model()` — either the file itself or a directory containing it
2. the `NER_MODEL_DATA_DIR` environment variable
3. `~/data/ner_model/`
4. `data/` inside this package

Each file is resolved independently, so keeping the weights in `~/data/ner_model/`
while `vocab.slot` stays in the package works. If the weights are missing, the
error names every location that was searched.

## Build

```bash
ln -s /home/amigo/ros/noetic/repos/github.com/tue-robotics/ner_model/ /home/amigo/ros/noetic/system/src/ner_model
tue-make ner_model
```

## Test with GPSR challenge (full robot stack)

Terminal 1 — start the robot simulator:
```bash
hero-start
```

Terminal 2 — start free mode:
```bash
hero-free-mode
```

Terminal 3 — (optional) launch RViz:
```bash
hero-rviz
```

Terminal 4 — run the GPSR challenge:
```bash
rosrun challenge_gpsr gpsr.py _robot_name:=hero _test_mode:=true _skip:=true


rosnode kill /hero/hmi/random_answerer
```

Then when it says "state your command":
```bash
rostopic pub /hero/hmi/string std_msgs/String "data: 'get the coke from the dining table'" --once
```

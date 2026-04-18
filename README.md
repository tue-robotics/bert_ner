# ner_model

BERT-based NER model for slot filling, integrated as a catkin package to replace the CFG grammar parser in the HMI pipeline.

## Prerequisites

- ROS Noetic workspace with `hmi`, `action_server`, and `tue_robocup` packages
- Python dependencies: `torch`, `transformers`
- Model files: `data/model.pth` and `data/vocab.slot` (not tracked in git, must be obtained separately)

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

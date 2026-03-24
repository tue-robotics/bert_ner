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

## Test with string_topic_answerer

Terminal 1:
```bash
roscore
```

Terminal 2:
```bash
rosrun hmi string_topic_answerer
```

Terminal 3 — send a goal then a sentence:
```bash
rostopic pub /string_topic_server/goal hmi_msgs/QueryActionGoal "goal:
  description: 'what can I do for you'
  grammar: 'dummy'
  target: 'T'" --once
```

```bash
rostopic pub /string std_msgs/String "data: 'go to the kitchen and find the beer'" --once
```

Check Terminal 2 for NER output and mapped semantics.

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
```

Terminal 5 — when the robot says "trigger me by stating my name":
```bash
rostopic pub /hero/hmi/string std_msgs/String "data: 'hero'" --once
```

Then when it says "state your command":
```bash
rostopic pub /hero/hmi/string std_msgs/String "data: 'go to the kitchen and find the beer'" --once
```

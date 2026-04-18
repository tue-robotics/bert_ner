# Extending the Robot's Vocabulary: Locations, Objects, and Actions

This guide explains how to add new locations, objects, person names, and actions to the HERO robot's NER-based command pipeline. There are **four layers** that must stay in sync for a new entity or action to work end-to-end.

---

## Architecture Overview

When you say "go to the kitchen and find the beer", this is what happens:

```
Voice/Text Input
      │
      ▼
┌─────────────────┐
│  NER Model       │  Tokenizes + slot-tags: "kitchen" → B-Location, "beer" → B-Object
│  (vocab.slot)    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Mapper          │  Converts slots → semantics dict:
│  (mapper.py)     │  {"action": "navigate-to", "target-location": {"id": "kitchen"}}
└────────┬────────┘
         ▼
┌─────────────────┐
│  Action Server   │  Looks up "kitchen" in world model, plans navigation
│  (action_server) │
└────────┬────────┘
         ▼
┌─────────────────┐
│  World Model     │  ED knows where "kitchen" is in the map
│  (ED + knowledge)│  common.py knows what objects exist
└────────┬────────┘
         ▼
      Robot moves
```

Each layer has its own configuration. All four must agree on the vocabulary.

---

## The Four Layers

### Layer 1: NER Model (`vocab.slot`)

**File:** `ner_model/data/vocab.slot`

This file defines what **slot types** the model can predict. It does NOT contain specific entity names (like "kitchen" or "beer") — it contains BIO tag labels like `B-Location`, `B-Object`, `B-Action-navigate-to`.

**Current slot types:**

| Slot Type | Purpose |
|-----------|---------|
| `B/I-Action-navigate-to` | Go to a place |
| `B/I-Action-find` | Find an object or person |
| `B/I-Action-hand-over` | Bring something to someone |
| `B/I-Action-pick-up` | Grab an object |
| `B/I-Action-place` | Put an object down |
| `B/I-Action-guide` | Lead a person somewhere |
| `B/I-Action-follow` | Follow a person |
| `B/I-Action-say` | Say something |
| `B/I-Action-answer-question` | Answer a question |
| `B/I-Action-count-and-tell` | Count objects and report |
| `B/I-Action-look-at` | Look at something |
| `B/I-Action-point-target` | Point at something |
| `B/I-Action-tell-name-of-person` | Identify a person |
| `B/I-Action-turn-toward-sound` | Turn toward a sound |
| `B/I-Object` | Any graspable object (coke, beer, cup, etc.) |
| `B/I-Location` | A room or furniture location |
| `B/I-Person` | A person's name |
| `B/I-SourceLocation` | Where to pick up from |
| `B/I-TargetLocation` | Where to place/deliver to |
| `B/I-Area` | An area within a location |
| `O` | Not an entity (filler word) |

**To add a new action type:**
1. Add `B-Action-<name>` and `I-Action-<name>` lines to `vocab.slot`
2. **Retrain the model** with training data that includes the new action
3. Update the mapper (Layer 2) to handle the new action

**To add new entity types** (e.g., `B-Color`, `B-Quantity`):
1. Add `B-<TypeName>` and `I-<TypeName>` to `vocab.slot`
2. **Retrain the model** with annotated training data
3. Update the mapper to convert the new type into semantics

> **Important:** The NER model recognizes entities by learned patterns, not by explicit name lists. If the model was trained on sentences containing "kitchen" and "living room" as locations, it will generalize to tag similar words as locations. Adding a new **specific** location name (e.g., "garage") does NOT require changing `vocab.slot` — it only requires that the model has seen enough similar examples during training to generalize.

---

### Layer 2: Mapper (`mapper.py`)

**File:** `ner_model/src/ner_model/mapper.py`

The mapper converts raw NER slot predictions into the structured semantics dictionary that the action server expects.

**Current entity-to-semantics mapping:**

```python
ENTITY_SLOT_TO_KEY = {
    "Object":         lambda text: {"object": {"type": normalize_entity(text)}},
    "SourceLocation": lambda text: {"source-location": {"id": normalize_entity(text)}},
    "TargetLocation": lambda text: {"target-location": {"id": normalize_entity(text)}},
    "Location":       lambda text: {"target-location": {"id": normalize_entity(text)}},
    "Person":         lambda text: {"target-location": {"type": "person", "id": normalize_entity(text)}},
    "Area":           lambda text: {"object": {"type": normalize_entity(text)}},
}
```

**How actions are mapped:**
Any span labeled `Action-X` becomes `{"action": "X"}`. For example:
- `Action-navigate-to` → `{"action": "navigate-to"}`
- `Action-find` → `{"action": "find"}`
- `Action-hand-over` → `{"action": "hand-over"}`

**To add a new entity type:**
Add an entry to `ENTITY_SLOT_TO_KEY`:

```python
"Color": lambda text: {"object-color": normalize_entity(text)},
```

**To add a new action type:**
No change needed in the mapper — any `Action-X` label is automatically mapped to `{"action": "X"}`. Just make sure the action name matches what the action server expects (Layer 3).

---

### Layer 3: Action Server

**File:** `action_server/action_server/src/action_server/actions/`

The action server receives the semantics dictionary and executes it. Each action is a Python class. The action name in the semantics dict is converted from `CamelCase` class names to `dash-case`:

| Class Name | Action Name in Semantics |
|------------|--------------------------|
| `NavigateTo` | `navigate-to` |
| `Find` | `find` |
| `HandOver` | `hand-over` |
| `PickUp` | `pick-up` |
| `Place` | `place` |
| `Guide` | `guide` |
| `Follow` | `follow` |
| `Say` | `say` |
| `AnswerQuestion` | `answer-question` |
| `CountAndTell` | `count-and-tell` |
| `LookAt` | `look-at` |
| `PointTarget` | `point-target` |
| `TellNameOfPerson` | `tell-name-of-person` |
| `TurnTowardSound` | `turn-toward-sound` |

**To add a new action:**
1. Create a new Python file in `action_server/src/action_server/actions/` (e.g., `wave.py`)
2. Define a class that extends `Action` (e.g., `class Wave(Action):`)
3. Implement `configure()`, `start()`, `_execute()`, and `_cancel()` methods
4. Import it in `actions/__init__.py`
5. The action factory will auto-discover it and register it as `wave`

**Semantics format expected by each action:**

```python
# navigate-to
{"action": "navigate-to", "target-location": {"id": "kitchen"}}

# find
{"action": "find", "object": {"type": "beer"}}
{"action": "find", "object": {"type": "person"}, "source-location": {"id": "living_room"}}

# hand-over (bring to a person)
{"action": "hand-over", "object": {"type": "coke"}, "target-location": {"type": "person", "id": "operator"}}

# pick-up
{"action": "pick-up", "object": {"type": "cup"}, "source-location": {"id": "dinner_table"}}

# place
{"action": "place", "object": {"type": "cup"}, "target-location": {"id": "kitchen_cabinet"}}

# say
{"action": "say", "sentence": "time"}

# guide
{"action": "guide", "target-location": {"id": "kitchen"}, "object": {"type": "person", "id": "john"}}
```

---

### Layer 4: Knowledge Base & World Model

**Knowledge file:** `tue_robocup/robocup_knowledge/src/robocup_knowledge/environments/impuls/common.py`

This file is selected at runtime based on the `ROBOT_ENV` environment variable (currently `impuls`). It defines everything the robot "knows" about its environment.

#### Adding a New Location

1. **Add to knowledge** (`common.py`):

```python
# In the locations list, add:
{'name': 'garage', 'room': 'garage', 'category': 'utility', 'manipulation': 'no'},
```

If it's a new room, also add it to the environment. The `location_rooms` list is auto-derived.

2. **Add to world model (ED):**

The robot uses ED (Environment Description) to know where things physically are. Location entities must exist in the ED world model so the robot can navigate to them.

**ED model config:** `hero_bringup/parameters/world_modeling/models_impuls.yaml`

```yaml
models:
  - id: garage
    type: some_model_type
```

The actual 3D model and pose must also be defined in the ED models directory.

3. **Add to Gazebo** (if the location should be visible in simulation):

The Gazebo world file must include the physical geometry. This is typically in the `.world` SDF file loaded by `hero-start`.

#### Adding a New Object

1. **Add to knowledge** (`common.py`):

```python
# In the objects list, add:
{'category': 'drink', 'name': 'juice', 'color': 'orange', 'volume': 376, 'weight': 335},
```

2. **Add to Gazebo** (for simulation): The object needs a Gazebo SDF model so it can be physically present and graspable in the simulation.

3. **No NER model change needed** if the model generalizes well to the new word. Since "juice" is similar to other drink words it was trained on, the model should tag it as `B-Object`. If not, retrain with examples containing the new object name.

#### Adding a New Person Name

1. **Add to knowledge** (`common.py`):

```python
female_names = ["anna", "beth", ..., "emily"]  # add "emily"
# or
male_names = ["alfred", "charles", ..., "kevin"]  # add "kevin"
```

2. **No NER model change needed** — person names are tagged as `B-Person` by the model based on context ("tell X", "bring to X", "find X"), not by memorizing specific names. New names should work if they appear in a similar context.

---

## Quick Reference: Adding Common Things

### "I want the robot to recognize a new object (e.g., 'laptop')"

| Layer | Change Needed | File |
|-------|--------------|------|
| NER Model | None (if model generalizes) or retrain | `ner_model/data/vocab.slot` (only if new entity type) |
| Mapper | None | — |
| Action Server | None | — |
| Knowledge | Add to `objects` list | `robocup_knowledge/.../impuls/common.py` |
| Gazebo | Add SDF model (for sim) | Gazebo world files |

### "I want the robot to go to a new room (e.g., 'garage')"

| Layer | Change Needed | File |
|-------|--------------|------|
| NER Model | None (if model generalizes) or retrain | — |
| Mapper | None | — |
| Action Server | None | — |
| Knowledge | Add to `locations` list | `robocup_knowledge/.../impuls/common.py` |
=| World Model | Add ED entity with pose | `hero_bringup/parameters/world_modeling/models_impuls.yaml` + ED models |
| Gazebo | Add room geometry (for sim) | Gazebo world files |

### "I want the robot to perform a new action (e.g., 'wave')"

| Layer | Change Needed | File |
|-------|--------------|------|
| NER Model | Add `B/I-Action-wave` to vocab, retrain | `ner_model/data/vocab.slot` |
| Mapper | None (auto-mapped) | — |
| Action Server | Create `Wave` action class | `action_server/.../actions/wave.py` |
| Knowledge | None | — |

---

## Testing After Changes

### Quick test (no Gazebo needed):

```bash
# Terminal 1
roscore

# Terminal 2
rosrun hmi string_topic_answerer

# Terminal 3: send a goal
rostopic pub /string_topic_server/goal hmi_msgs/QueryActionGoal "goal:
  description: 'test'
  grammar: 'dummy'
  target: 'T'" --once

# Terminal 4: send your command
rostopic pub /string std_msgs/String "data: 'go to the garage and find the laptop'" --once

# Terminal 5: check the result
rostopic echo /string_topic_server/result -n 1
```

### Full end-to-end test (with Gazebo):

```bash
# Terminal 1: Start robot simulator
hero-start

# Terminal 2: Start navigation + action server
roslaunch hero_bringup free_mode.launch

# Terminal 3: Kill random answerer
rosnode kill /hero/hmi/random_answerer

# Terminal 5: Run GPSR challenge
rosrun challenge_gpsr gpsr.py _robot_name:=hero _test_mode:=true _skip:=true

# Terminal 5: Command the robot to do something e.g.: 
rostopic pub /hero/hmi/string std_msgs/String "data: 'get the coke from the dinner table'" --once

```

---

## File Reference

| File | Purpose |
|------|---------|
| `ner_model/data/vocab.slot` | BIO slot labels the NER model can predict |
| `ner_model/data/model.pth` | Trained model weights |
| `ner_model/src/ner_model/mapper.py` | Converts NER output → action server semantics |
| `ner_model/src/ner_model/parser.py` | Loads model, runs inference, calls mapper |
| `robocup_knowledge/.../impuls/common.py` | Locations, objects, names for the impuls environment |
| `robocup_knowledge/.../impuls/challenge_gpsr.py` | GPSR grammar (used by old CFG parser, kept for reference) |
| `action_server/.../actions/*.py` | Action implementations the robot can execute |
| `hero_bringup/parameters/world_modeling/` | ED world model config (furniture models + poses) |
| `conversation_engine/.../engine.py` | Conversation flow: receives HMI result, sends to action server |
| `hmi/.../common.py` | `parse_sentence()` — where NERParser replaces CFGParser |

# GPSR test commands — `ROBOT_ENV=impuls`

Copy-paste commands for manual GPSR + entity-linker testing.  
Requires: GPSR running, `ROBOT_ENV=impuls`, linker enabled (`sentence-transformers` in venv).

**Watch logs for:**
```
Entity linker: linked '...' -> '...' (score=...)
Entity mapper: built semantics ...
```

---

## Commands (copy-paste)

1. Bring me a coke from the dining table
2. Pick up the fanta from the dining room
3. Get the water from the living room table
4. Grab the mentos from the office shelf
5. Take the plate from the work bench in the office
6. Pick up the sponge from the cabinet and put it on the salon table
7. Go to the office and find the pringles
8. Get the apple from the dinner table and bring it to me
9. Find Jennifer in the dining room
10. Bring a coke from the dining table to Daniel at the salon table

---

## Per-command notes

### 1. Bring me a coke from the dining table
- **Tests:** drink object + dining-table → `dinner_table` synonym
- **Expected links:** `coke`, `dinner_table`
- **Sim:** coke spawned in `objects_sdf.yaml` ✓

### 2. Pick up the fanta from the dining room
- **Tests:** room vs furniture disambiguation (`dining_room` vs `dinner_table`)
- **Expected links:** `fanta`; location may be `dining_room` or `dinner_table` — check linker scores
- **Note:** drinks are on `dinner_table` per `common.py`, not the room

### 3. Get the water from the living room table
- **Tests:** `living room table` → `salon_table`
- **Expected links:** `water`, `salon_table`

### 4. Grab the mentos from the office shelf
- **Tests:** `office shelf` → `closet` (snacks live in closet per `category_locations`)
- **Expected links:** `mentos`, `closet`

### 5. Take the plate from the work bench in the office
- **Tests:** ambiguous furniture — `work_bench1` vs `work_bench2`
- **Expected links:** `plate`, one of `work_bench1` / `work_bench2`
- **Watch:** top-2 linker scores if close

### 6. Pick up the sponge from the cabinet and put it on the salon table
- **Tests:** multi-action (pick-up + place), split wordpiece `sponge`, place `reference`, Area/`on` ignored
- **Expected semantics:**
  ```python
  pick-up: sponge from cabinet
  place:   reference → salon_table
  ```
- **Sim caveat:** sponge **not** in `objects_sdf.yaml` — language pipeline works, Find may fail until sponge is spawned

### 7. Go to the office and find the pringles
- **Tests:** multi-action navigate + find
- **Expected links:** `office`, `pringles` (in `closet` per knowledge)

### 8. Get the apple from the dinner table and bring it to me
- **Tests:** pick-up + hand-over, `dining table` / `dinner table` → `dinner_table`
- **Expected links:** `apple`, `dinner_table`, operator/me
- **Sim:** apple spawned ✓ — **validated end-to-end**

### 9. Find Jennifer in the dining room
- **Tests:** person name + room
- **Expected links:** `jennifer`, `dining_room`

### 10. Bring a coke from the dining table to Daniel at the salon table
- **Tests:** hardest — 4 entities, hand-over with person + two locations
- **Expected links:** `coke`, `dinner_table`, `daniel`, `salon_table`

---

## Quick reference: natural phrase → canonical ID

| You say | Links to |
|---------|----------|
| dining table / dinner table | `dinner_table` |
| living room table | `salon_table` |
| office shelf | `closet` |
| book shelf | `bookcase` |

## Where objects should be (`impuls/common.py`)

| Category | Default location |
|----------|------------------|
| drink | `dinner_table` |
| food, snack | `closet` |
| cleaning_stuff, cutlery, container | `cabinet` |

## Actually spawned in sim (`objects_sdf.yaml`)

Currently: **coke**, **apple** only.  
Commands involving sponge, fanta, mentos, etc. may pass language grounding but fail at Find if not spawned/perceived.

---

## Suggested test order

**Easy (sim-friendly):** 1, 8  
**Linker stress:** 3, 4, 10  
**Multi-action:** 6, 7, 8  
**Ambiguous:** 2, 5

---

## Pass / fail checklist

- [ ] `Entity linker: ready` on first command (cold start ~2s)
- [ ] No `failed to initialize: sentence-transformers`
- [ ] `Entity linker: linked` or `exact match` for each mention
- [ ] Score > 0.60 (or sensible exact match)
- [ ] `Entity mapper: built semantics` looks correct
- [ ] Action server does not ask for missing `actions[0].object`
- [ ] Task succeeds or fails only on sim/perception (not language)

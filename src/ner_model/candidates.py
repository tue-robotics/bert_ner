import os
from typing import List

import rospy

from .linker import EntityCandidate


def _candidate_text(entity_id: str, description: str = "") -> str:
    """
    Candidates are the entities the robot knows about in the current environment:
    the objects, locations and person names in robocup_knowledge for ROBOT_ENV.

    The entity linker matches each mention the NER model extracts (e.g. "dining
    table") to the closest candidate, so the action server gets an id that exists
    in the world model (e.g. "dinner_table"). Each candidate is an EntityCandidate:

    - entity_id: the world-model id, e.g. "dinner_table"
    - label: Object, Location or Person; mentions are only compared to candidates
    with the same label
    - text: what the linker compares mentions against, e.g. "dinner table, dining room"
    
    Build a candidate's text: the id in plain words, plus its context if any.

    e.g. ("dinner_table", "dining room") -> "dinner table, dining room"
    """
    text = entity_id.replace("_", " ")
    if description:
        return "{}, {}".format(text, description)
    return text


def candidates_from_common(common) -> List[EntityCandidate]:
    """Build linker candidates from robocup_knowledge common module."""
    candidates = []

    for name in getattr(common, "object_names", []):
        category = ""
        for obj in getattr(common, "objects", []):
            if obj.get("name") == name:
                category = obj.get("category", "")
                break
        description = category.replace("_", " ") if category else ""
        candidates.append(
            EntityCandidate(
                entity_id=name,
                label="Object",
                text=_candidate_text(name, description),
            )
        )

    location_ids = set(getattr(common, "location_names", []))
    for room in getattr(common, "location_rooms", []):
        location_ids.add(room)

    for loc in getattr(common, "locations", []):
        location_ids.add(loc["name"])

    for name in sorted(location_ids):
        room = ""
        for loc in getattr(common, "locations", []):
            if loc.get("name") == name:
                room = loc.get("room", "")
                break
        description = room.replace("_", " ") if room else ""
        candidates.append(
            EntityCandidate(
                entity_id=name,
                label="Location",
                text=_candidate_text(name, description),
            )
        )

    for name in getattr(common, "names", []):
        candidates.append(
            EntityCandidate(
                entity_id=name,
                label="Person",
                text=name,
            )
        )

    return candidates


def load_candidates() -> List[EntityCandidate]:
    """
    Load entity candidates for the active environment.

    Uses ROBOT_ENV + robocup_knowledge/common.py when available.
    """
    robot_env = os.environ.get("ROBOT_ENV")
    if not robot_env:
        rospy.logwarn(
            "Entity linker: ROBOT_ENV not set — no world-model candidates loaded"
        )
        return []

    try:
        from robocup_knowledge import knowledge_loader

        common = knowledge_loader.load_knowledge("common")
        candidates = candidates_from_common(common)
        rospy.loginfo(
            "Entity linker: loaded %d candidates from ROBOT_ENV='%s' (common)",
            len(candidates),
            robot_env,
        )
        return candidates
    except Exception as exc:
        rospy.logwarn(
            "Entity linker: failed to load candidates for ROBOT_ENV='%s': %s",
            robot_env,
            exc,
        )
        return []

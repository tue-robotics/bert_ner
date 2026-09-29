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
    # name -> description lookups; reversed() so the first entry per name wins
    categories = {
        obj.get("name"): obj.get("category", "")
        for obj in reversed(getattr(common, "objects", []))
    }
    rooms = {
        loc["name"]: loc.get("room", "")
        for loc in reversed(getattr(common, "locations", []))
    }
    location_ids = set().union(
        getattr(common, "location_names", []), getattr(common, "location_rooms", []), rooms
    )

    sources = [
        ("Object", getattr(common, "object_names", []), categories),
        ("Location", sorted(location_ids), rooms),
        ("Person", getattr(common, "names", []), {}),
    ]

    return [
        EntityCandidate(
            entity_id=name,
            label=label,
            text=_candidate_text(name, (descriptions.get(name) or "").replace("_", " ")),
        )
        for label, names, descriptions in sources
        for name in names
    ]


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

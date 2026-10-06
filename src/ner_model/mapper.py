import re
from typing import Any, Dict, List, Optional

import rospy

from .linker import BiEncoderLinker

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def extract_spans(predictions):
    """
    Merge BIO-tagged token+slot pairs into contiguous spans.

    Input:  [{"token": "go", "slot": "B-Action-navigate-to"},
             {"token": "to", "slot": "I-Action-navigate-to"}, ...]
    Output: [("Action-navigate-to", "go to"), ("Location", "kitchen"), ...]
    """
    spans = []
    current_label = None
    current_tokens = []
    orphan_prefix = None

    def _close_span():
        nonlocal current_label, current_tokens, orphan_prefix
        if current_label is not None:
            spans.append((current_label, " ".join(current_tokens)))
            current_label = None
            current_tokens = []
        orphan_prefix = None

    for pred in predictions:
        token = pred["token"]
        slot = pred["slot"]

        if token in SPECIAL_TOKENS or slot == "[PAD]":
            continue

        if token.startswith("##"):
            piece = token[2:]
            if current_tokens:
                current_tokens[-1] += piece
                continue
            if slot == "O":
                continue
            prefix = slot[0]
            label = slot[2:]
            if prefix == "B":
                saved_orphan = orphan_prefix
                _close_span()
                current_label = label
                current_tokens = [(saved_orphan or "") + piece]
            elif prefix == "I" and current_label == label:
                current_tokens.append(piece)
            continue

        if slot == "O" and token == "_" and current_label is not None:
            current_tokens.append(token)
            orphan_prefix = None
            continue

        if slot == "O":
            _close_span()
            if len(token) == 1 and token.isalpha():
                orphan_prefix = token
            else:
                orphan_prefix = None
            continue

        orphan_prefix = None
        prefix = slot[0]
        label = slot[2:]

        if prefix == "B":
            _close_span()
            current_label = label
            current_tokens = [token]
        elif prefix == "I" and current_label is not None:
            current_tokens.append(token)

    if current_label is not None:
        spans.append((current_label, " ".join(current_tokens)))

    rospy.logdebug("Entity mapper: extracted %d spans: %s", len(spans), spans)
    return spans


def normalize_entity(text):
    """
    Lowercase, collapse whitespace around underscores, and replace spaces with underscores.
    e.g. "living room" -> "living_room"
         "Dinner Table" -> "dinner_table"
         "kitchen _ cabinet" -> "kitchen_cabinet"
    """
    text = text.lower().strip()
    text = re.sub(r"\s*_\s*", "_", text)
    return text.replace(" ", "_")


def _resolve_entity_id(
    slot_label: str, mention_text: str, linker: Optional[BiEncoderLinker]
) -> str:
    if linker is None:
        canonical_id = normalize_entity(mention_text)
        rospy.logdebug(
            "Entity mapper: no linker — normalized '%s' (%s) -> '%s'",
            mention_text,
            slot_label,
            canonical_id,
        )
        return canonical_id

    try:
        result = linker.link(slot_label, mention_text)
    except Exception as exc:
        # Linking is an enhancement; a failure should degrade to plain
        # normalization rather than lose the whole command.
        rospy.logwarn(
            "Entity mapper: linker raised on '%s' (%s): %s: %s",
            mention_text,
            slot_label,
            type(exc).__name__,
            exc,
        )
        result = None

    if result is not None:
        return result.entity_id

    fallback_id = normalize_entity(mention_text)
    rospy.logwarn(
        "Entity mapper: linking failed for '%s' (%s) — falling back to '%s'",
        mention_text,
        slot_label,
        fallback_id,
    )
    return fallback_id


def _entity_dict_for_slot(
    slot_label: str, mention_text: str, linker: Optional[BiEncoderLinker]
) -> Dict[str, Any]:
    canonical_id = _resolve_entity_id(slot_label, mention_text, linker)

    if slot_label == "Object":
        return {"object": {"type": canonical_id}}
    if slot_label == "SourceLocation":
        return {"source-location": {"id": canonical_id}}
    if slot_label == "TargetLocation":
        return {"target-location": {"id": canonical_id}}
    if slot_label == "Location":
        return {"target-location": {"id": canonical_id}}
    if slot_label == "Person":
        return {"target-location": {"type": "person", "id": canonical_id}}

    rospy.logwarn("Entity mapper: unhandled slot '%s'", slot_label)
    return {}


ENTITY_SLOTS = {"Object", "SourceLocation", "TargetLocation", "Location", "Person"}
SKIPPED_SLOTS = {"Area"}


def build_semantics(
    predictions, linker: Optional[BiEncoderLinker] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert raw NER predictions into the action server semantics dict.

    Groups entity spans under their preceding action span.
    Returns: {"actions": [{"action": "navigate-to", ...}, ...]}
    """
    spans = extract_spans(predictions)

    actions = []
    current_action = None

    for label, text in spans:
        if label.startswith("Action-"):
            if current_action is not None:
                actions.append(current_action)
            action_name = label[len("Action-"):]
            current_action = {"action": action_name}
            rospy.logdebug("Entity mapper: new action '%s'", action_name)

        elif label in ENTITY_SLOTS:
            if current_action is None:
                current_action = {"action": "unknown"}
                rospy.logwarn(
                    "Entity mapper: entity '%s' before any action — using action=unknown",
                    text,
                )
            entity_dict = _entity_dict_for_slot(label, text, linker)
            rospy.loginfo(
                "Entity mapper: slot '%s' mention '%s' -> %s",
                label,
                text,
                entity_dict,
            )
            current_action.update(entity_dict)
        elif label in SKIPPED_SLOTS:
            rospy.logdebug(
                "Entity mapper: skipping slot '%s' mention '%s'",
                label,
                text,
            )

    if current_action is not None:
        actions.append(current_action)

    for action in actions:
        if action.get("action") == "place" and "object" not in action:
            action["object"] = {"type": "reference"}
            rospy.loginfo(
                "Entity mapper: inferred place reference object for action %s",
                action,
            )

    if not actions:
        rospy.logwarn("Entity mapper: no actions parsed from predictions")
        return {"actions": [{"action": "unknown"}]}

    semantics = {"actions": actions}
    rospy.loginfo("Entity mapper: built semantics %s", semantics)
    return semantics

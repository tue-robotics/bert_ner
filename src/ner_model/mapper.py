import re

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

    for pred in predictions:
        token = pred["token"]
        slot = pred["slot"]

        if token in SPECIAL_TOKENS or slot == "[PAD]":
            continue

        if token.startswith("##"):
            if current_tokens:
                current_tokens[-1] += token[2:]
            continue

        # Bridge underscores inside a multi-token entity span.
        # BERT tokenizers split "kitchen_cabinet" into "kitchen", "_", "cabinet".
        # If we're mid-span and see an "O"-tagged "_", treat it as part of the span.
        if slot == "O" and token == "_" and current_label is not None:
            current_tokens.append(token)
            continue

        if slot == "O":
            if current_label is not None:
                spans.append((current_label, " ".join(current_tokens)))
                current_label = None
                current_tokens = []
            continue

        prefix = slot[0]
        label = slot[2:]

        if prefix == "B":
            if current_label is not None:
                spans.append((current_label, " ".join(current_tokens)))
            current_label = label
            current_tokens = [token]
        elif prefix == "I" and current_label is not None:
            current_tokens.append(token)

    if current_label is not None:
        spans.append((current_label, " ".join(current_tokens)))

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


ENTITY_SLOT_TO_KEY = {
    "Object":         lambda text: {"object": {"type": normalize_entity(text)}},
    "SourceLocation": lambda text: {"source-location": {"id": normalize_entity(text)}},
    "TargetLocation": lambda text: {"target-location": {"id": normalize_entity(text)}},
    "Location":       lambda text: {"target-location": {"id": normalize_entity(text)}},
    "Person":         lambda text: {"target-location": {"type": "person", "id": normalize_entity(text)}},
    "Area":           lambda text: {"object": {"type": normalize_entity(text)}},
}


def build_semantics(predictions):
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

        elif label in ENTITY_SLOT_TO_KEY:
            if current_action is None:
                current_action = {"action": "unknown"}
            entity_dict = ENTITY_SLOT_TO_KEY[label](text)
            current_action.update(entity_dict)

    if current_action is not None:
        actions.append(current_action)

    if not actions:
        return {"actions": [{"action": "unknown"}]}

    return {"actions": actions}

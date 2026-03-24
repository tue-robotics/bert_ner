import logging
import os
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.path.dirname(__file__)).parent.parent / "data"


def _load_slot_names():
    vocab_path = DATA_DIR / "vocab.slot"
    names = ["[PAD]"]
    names += vocab_path.read_text("utf-8").strip().splitlines()
    return names


class JointIntentAndSlotFillingModel(nn.Module):
    slot_names = _load_slot_names()
    slot_map = {label: i for i, label in enumerate(slot_names)}

    def __init__(self, slot_num_labels, model_name="bert-base-cased", dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, slot_num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        return slot_logits


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    slot_num_labels = len(JointIntentAndSlotFillingModel.slot_map)
    model = JointIntentAndSlotFillingModel(slot_num_labels=slot_num_labels)

    model_path = DATA_DIR / "model.pth"
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded on %s", device)

    return model, tokenizer, device

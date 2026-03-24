import logging
import torch
import torch.nn.functional as F
from .model import JointIntentAndSlotFillingModel

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.idx_to_slot = {v: k for k, v in JointIntentAndSlotFillingModel.slot_map.items()}

    def predict(self, text):
        encoded = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=45
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            slot_logits = self.model(
                encoded["input_ids"], attention_mask=encoded["attention_mask"]
            )

        slot_probs = F.softmax(slot_logits, dim=-1)
        slot_predictions = torch.argmax(slot_probs, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].cpu().numpy())
        predicted_slots = [self.idx_to_slot[idx] for idx in slot_predictions[0].cpu().numpy()]

        return [{"token": tok, "slot": slot} for tok, slot in zip(tokens, predicted_slots)]

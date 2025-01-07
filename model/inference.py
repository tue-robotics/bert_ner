import torch
from transformers import AutoModel
from torch import nn
from pathlib import Path
from transformers import BertTokenizer
import numpy as np
import os
import torch.nn.functional as F

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class JointIntentAndSlotFillingModel(nn.Module):
    def __init__(self, slot_num_labels, model_name="bert-base-cased", dropout_prob=0.1):
        super(JointIntentAndSlotFillingModel, self).__init__()
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.slot_classifier = nn.Linear(
            self.bert.config.hidden_size, slot_num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        return slot_logits


# Define the slot_map from the training script
slot_names = ["[PAD]"]  # Start with the PAD token
slot_names += Path("vocab.slot").read_text("utf-8").strip().splitlines()
slot_map = {label: i for i, label in enumerate(slot_names)}

# Recreate the model
slot_num_labels = len(slot_map)
model = JointIntentAndSlotFillingModel(slot_num_labels=slot_num_labels)
model.load_state_dict(torch.load("model.pth"))

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully.")

# Tokenize a sample input
sample_text = "Can you bring me my mobile phone from the table?"
encoded_input = tokenizer(sample_text, return_tensors="pt",
                          padding=True, truncation=True, max_length=45)
encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

# Perform inference
with torch.no_grad():
    slot_logits = model(
        encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"])

print("Slot logits:", slot_logits)

# Get the predictions
slot_probs = F.softmax(slot_logits, dim=-1)
slot_predictions = torch.argmax(slot_probs, dim=-1)

# Inverse slot_map for mapping indices back to slot labels
idx_to_slot = {v: k for k, v in slot_map.items()}

# Decode the tokens back to words and slot labels
input_ids = encoded_input["input_ids"][0].cpu().numpy()
tokens = tokenizer.convert_ids_to_tokens(input_ids)
predicted_slots = [idx_to_slot[idx]
                   for idx in slot_predictions[0].cpu().numpy()]

# Print tokens and their corresponding slot labels
output = []
for token, slot in zip(tokens, predicted_slots):
    if token not in ["[PAD]", "[CLS]", "[SEP]"]:  # Exclude special tokens
        output.append((token, slot))

print("Predictions:")
for token, slot in output:
    print(f"{token:10} -> {slot}")

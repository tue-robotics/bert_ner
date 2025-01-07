from transformers import BertConfig
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModel
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve
from torch.utils.data import Dataset
import torch
from transformers import Trainer, TrainingArguments
from transformers import BertForTokenClassification
from transformers import BertTokenizer
import numpy as np
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# try githubs approach: https://github.com/m2dsupsdlclass/lectures-labs/blob/master/labs/06_deep_nlp/Transformers_Joint_Intent_Classification_Slot_Filling_rendered.ipynb
# https://github.com/monologg/JointBERT/blob/master/data_loader.py
"""
SNIPS_DATA_BASE_URL = (
    "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
    "master/data/snips/"
)
for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:
    path = Path(filename)
    if not path.exists():
        print(f"Downloading {filename}...")
        urlretrieve(SNIPS_DATA_BASE_URL + filename + "?raw=true", path)
"""
lines_train = Path("train").read_text("utf-8").strip().splitlines()
print("########################################### lines_train:")
print(lines_train[:5])

""" ΨΘΔΑ ΤΕΣΤ
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple neural network


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Input layer
        self.fc2 = nn.Linear(50, 20)  # Hidden layer
        self.fc3 = nn.Linear(20, 1)   # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model and move it to the selected device (CPU or CUDA)
model = SimpleNN().to(device)

# Create random input and target tensors on the selected device
input_data = torch.randn(100, 10).to(device)  # 100 samples, 10 features
target_data = torch.randn(100, 1).to(device)  # 100 target values

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model for a few epochs
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, target_data)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("CUDA test completed.")
"""


def parse_line(line):
    utterance_data, intent_label = line.split(" <=> ")
    items = utterance_data.split()
    words = [item.rsplit(":", 1)[0]for item in items]
    word_labels = [item.rsplit(":", 1)[1]for item in items]
    return {
        "intent_label": intent_label,
        "words": " ".join(words),
        "word_labels": " ".join(word_labels),
        "length": len(words),
    }


print("########################################### Parsed example:")
print(parse_line(lines_train[0]))


parsed = [parse_line(line) for line in lines_train]

df_train = pd.DataFrame([p for p in parsed if p is not None])
print("########################################### df_train:")
print(df_train)

lines_valid = Path("valid").read_text("utf-8").strip().splitlines()
lines_test = Path("test").read_text("utf-8").strip().splitlines()

df_valid = pd.DataFrame([parse_line(line) for line in lines_valid])
df_test = pd.DataFrame([parse_line(line) for line in lines_test])
# print("########################################### df_valid:")
# print(df_valid)

"""
def encode_dataset(tokenizer, text_sequences, max_length):
    token_ids = np.zeros(shape=(len(text_sequences), max_length),
                         dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        # Ensure token IDs are within vocabulary range
        if max(encoded) >= tokenizer.vocab_size:
            print(
                f"Out-of-range token ID in input sequence '{text_sequence}': {encoded}")
            print('------------------------------------------------------------')
            encoded = [min(id, tokenizer.vocab_size - 1)
                       for id in encoded]  # Clip values
        token_ids[i, 0:len(encoded)] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_masks": attention_masks}
"""


def encode_dataset(tokenizer, text_sequences, max_length):
    inputs = tokenizer.batch_encode_plus(
        text_sequences,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
        # TODO: return_offsets_mapping=True  # Add this line since it helps track token positions relative to the original words
    )
    return {"input_ids": inputs["input_ids"], "attention_masks": inputs["attention_mask"]}


encoded_train = encode_dataset(tokenizer, df_train["words"], 45)
encoded_valid = encode_dataset(tokenizer, df_valid["words"], 45)
encoded_test = encode_dataset(tokenizer, df_test["words"], 45)
print("########################################### encoded_train_input_ids:")
print(encoded_train["input_ids"])
print("########################################### encoded_train_attention_masks:")
print(encoded_train["attention_masks"])

slot_names = ["[PAD]"]
slot_names += Path("vocab.slot").read_text("utf-8").strip().splitlines()
slot_map = {}
for label in slot_names:
    slot_map[label] = len(slot_map)
print("########################################### slot_map:")
print(slot_map)

# Create slots for each word in the text
"""
TODO: Implement the encode_token_labels function to mark subsequent subwords with a different placeholder (e.g., X).
for word, word_label in zip(text_sequence.split(), word_labels.split()):
    tokens = tokenizer.tokenize(word)
    encoded_labels.append(slot_map[word_label])  # First subword with correct label
    # Use 'X' or another placeholder for subsequent subwords
    encoded_labels.extend([slot_map.get('X', 0)] * (len(tokens) - 1))
"""


def encode_token_labels(text_sequences, slot_names, tokenizer, slot_map,
                        max_length):
    encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, (text_sequence, word_labels) in enumerate(
            zip(text_sequences, slot_names)):
        encoded_labels = []
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if not expand_label in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
    return encoded


slot_train = encode_token_labels(
    df_train["words"], df_train["word_labels"], tokenizer, slot_map, 45)
slot_valid = encode_token_labels(
    df_valid["words"], df_valid["word_labels"], tokenizer, slot_map, 45)
slot_test = encode_token_labels(
    df_test["words"], df_test["word_labels"], tokenizer, slot_map, 45)

print("########################################### slot_train:")
print(slot_train)
# Convert numpy arrays to tensors
input_ids_train = torch.tensor(encoded_train["input_ids"])
attention_masks_train = torch.tensor(encoded_train["attention_masks"])
slot_labels_train = torch.tensor(slot_train)
# Use the same process for validation and test datasets
input_ids_valid = torch.tensor(encoded_valid["input_ids"])
attention_masks_valid = torch.tensor(encoded_valid["attention_masks"])
slot_labels_valid = torch.tensor(slot_valid)
print("########################################### input_ids_train:")
print(input_ids_train)
print("########################################### attention_masks_train:")
print(attention_masks_train)
print("########################################### slot_labels_train:")
print(slot_labels_train)


def batch_data(input_ids, attention_masks, labels, batch_size):
    """
    This function creates a TensorDataset and DataLoader from input tensors.

    Args:
    - input_ids (torch.Tensor): Token IDs tensor.
    - attention_masks (torch.Tensor): Attention masks tensor.
    - labels (torch.Tensor): Slot labels tensor.
    - batch_size (int): Batch size.

    Returns:
    - DataLoader: DataLoader with padded and batched data.
    """
    # Create TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Initialize DataLoader with batch size and other configurations
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        drop_last=False,  # Ensure no data is dropped
    )

    return dataloader


# Create TensorDataset
# Use the function to create DataLoaders
train_dataloader = batch_data(
    input_ids_train, attention_masks_train, slot_labels_train, batch_size=32)
val_dataloader = batch_data(
    input_ids_valid, attention_masks_valid, slot_labels_valid, batch_size=32)


class JointIntentAndSlotFillingModel(nn.Module):
    def __init__(self, slot_num_labels, model_name="bert-base-cased", dropout_prob=0.1):
        super(JointIntentAndSlotFillingModel, self).__init__()
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        self.dropout = nn.Dropout(dropout_prob)

        # Intent and slot classifiers
        # self.intent_classifier = nn.Linear(
        # self.bert.config.hidden_size, intent_num_labels)
        self.slot_classifier = nn.Linear(
            self.bert.config.hidden_size, slot_num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get BERT outputs (sequence and pooled)
        # In the forward method:
        # print("########################################### Forward_pass:")
        # print("FORWARD input_ids:", input_ids.shape)
        # print("FORWARD attention_mask:", attention_mask.shape)
        # print("FORWARD Max token ID:", torch.max(input_ids))

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        # pooled_output = outputs.pooler_output

        # Slot filling using sequence_output
        sequence_output = self.dropout(sequence_output)
        slot_logits1 = self.slot_classifier(sequence_output)

        # Intent classification using pooled_output
        # pooled_output = self.dropout(pooled_output)
        # intent_logits = self.intent_classifier(pooled_output)

        return slot_logits1  # , intent_logits


# Define model and optimizer
config = BertConfig.from_pretrained("bert-base-cased")
config.return_dict = False
model = JointIntentAndSlotFillingModel(slot_num_labels=len(slot_map))
optimizer = AdamW(model.parameters(), lr=3e-5)

# Define loss functions for slot and intent
# TODO Set ignore_index in CrossEntropyLoss to the [PAD] index to avoid penalizing padding in the loss calculation.
# pad_token_label_id = slot_map["[PAD]"]
# slot_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_label_id)
slot_loss_fn = nn.CrossEntropyLoss()
# intent_loss_fn = nn.CrossEntropyLoss()

"""
# Move everything to CPU
device = torch.device("cpu")
model.to(device)

# Fetch a small batch
sample_input_ids, sample_attention_mask, sample_slot_labels = next(
    iter(train_dataloader))
sample_input_ids, sample_attention_mask, sample_slot_labels = (
    sample_input_ids.to(device),
    sample_attention_mask.to(device),
    sample_slot_labels.to(device),
)
sample_input_ids = sample_input_ids.long()  # Convert to int64
sample_attention_mask = sample_attention_mask.long()  # Convert to int64
sample_slot_labels = sample_slot_labels.long()  # Convert to int64

try:
    slot_logits = model(sample_input_ids, attention_mask=sample_attention_mask)
    print("Model output shape:", slot_logits.shape)
except Exception as e:
    print("Error during model forward pass on CPU:", e)
"""
# print(model)
# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"we still have :D {device}")
model.to(device)

for epoch in range(2):  # Set the number of epochs
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        # Unpack batch
        input_ids, attention_mask, slot_labels = [b for b in batch]
        input_ids = input_ids.type(torch.LongTensor).to(device)
        attention_mask = attention_mask.type(torch.LongTensor).to(device)
        slot_labels = slot_labels.type(torch.LongTensor).to(device)

        # Ensure the data types are correct
        # input_ids = input_ids.long()  # Convert to int64
        # attention_mask = attention_mask.long()  # Convert to int64
        # slot_labels = slot_labels.long()  # Convert to int64

        # Check if any token IDs exceed the tokenizer's vocabulary size
        max_vocab_index = tokenizer.vocab_size
        # assert (input_ids < max_vocab_index).all(
        # ), "One or more token IDs exceed the tokenizer's vocabulary size."

        # Ensure slot_labels are of type torch.long and check their validity
        # assert (slot_labels < len(slot_map)).all(
        # ), "Slot labels contain values exceeding slot_map range"

        # Debugging check for input_ids range
        # if torch.max(input_ids) >= tokenizer.vocab_size:
        # print("Found token ID out of range for input:", input_ids)
        # input_ids = torch.clamp(input_ids, max=tokenizer.vocab_size - 1)
        # try:
        # Forward pass
        # Check for any values outside the expected range
        # print("Max token ID:", torch.max(input_ids).item(),
        # "Expected max:", tokenizer.vocab_size - 1)

        slot_logits = model(input_ids, attention_mask=attention_mask)
        print("########################################### Forward_pass:")
        # print("Input IDs:", input_ids)
        # print("Slot logits:", slot_logits)
        # print("Slot labels:", slot_labels)
        # print("Slot logits shape:", slot_logits.shape)  # Debugging output
        # print("Embedding layer shape:",
        # model.bert.embeddings.word_embeddings.weight.shape)
        # vocab_size = tokenizer.vocab_size
        # print(f"Tokenizer vocabulary size: {vocab_size}")
        # Get the vocabulary size from the BERT model's configuration
        vocab_size1 = model.bert.config.vocab_size
        # print(f"Model vocabulary size: {vocab_size1}")
        # Compute losses
        slot_loss = slot_loss_fn(
            slot_logits.view(-1,
                             slot_logits.shape[-1]), slot_labels.view(-1)
        )
        loss = slot_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{2}, Loss: {total_loss / len(train_dataloader)}")

torch.save(model.state_dict(), "model.pth")

from transformers import BertConfig
from torch.optim import AdamW
from torch import nn
import pandas as pd
from pathlib import Path
import torch
from transformers import BertTokenizer
import os
import model.modeling_bert
import utils.data_utils
import utils.trainer
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# try githubs approach: https://github.com/m2dsupsdlclass/lectures-labs/blob/master/labs/06_deep_nlp/Transformers_Joint_Intent_Classification_Slot_Filling_rendered.ipynb
# https://github.com/monologg/JointBERT/blob/master/data_loader.py

# Load the training data
lines_train = Path("data/train").read_text("utf-8").strip().splitlines()
lines_valid = Path("data/valid").read_text("utf-8").strip().splitlines()
lines_test = Path("data/test").read_text("utf-8").strip().splitlines()


dataset = utils.data_utils.Dataset()
print(dataset.parse_line(lines_train[0]))

# Parse all examples and Convert the parsed data to a DataFrame
df_train = pd.DataFrame([dataset.parse_line(line) for line in lines_train])
df_valid = pd.DataFrame([dataset.parse_line(line) for line in lines_valid])
df_test = pd.DataFrame([dataset.parse_line(line) for line in lines_test])

# Encode the datasets
encoded_train = dataset.encode_dataset(tokenizer, df_train["words"], 45)
encoded_valid = dataset.encode_dataset(tokenizer, df_valid["words"], 45)
encoded_test = dataset.encode_dataset(tokenizer, df_test["words"], 45)

# Define the slot_map
slot_names = ["[PAD]"]
slot_names += Path("data/vocab.slot").read_text("utf-8").strip().splitlines()
slot_map = {}
# Create a mapping from slot names to integers
for label in slot_names:
    slot_map[label] = len(slot_map)

# Encode the slot labels
slot_train = dataset.encode_token_labels(
    df_train["words"], df_train["word_labels"], tokenizer, slot_map, 45)
slot_valid = dataset.encode_token_labels(
    df_valid["words"], df_valid["word_labels"], tokenizer, slot_map, 45)
slot_test = dataset.encode_token_labels(
    df_test["words"], df_test["word_labels"], tokenizer, slot_map, 45)
 
 
# Convert numpy arrays to tensors
input_ids_train = torch.tensor(encoded_train["input_ids"])
attention_masks_train = torch.tensor(encoded_train["attention_masks"])
slot_labels_train = torch.tensor(slot_train)
# Use the same process for validation and test datasets
input_ids_valid = torch.tensor(encoded_valid["input_ids"])
attention_masks_valid = torch.tensor(encoded_valid["attention_masks"])
slot_labels_valid = torch.tensor(slot_valid)

# Create TensorDataset
# Use the function to create DataLoaders
train_dataloader = dataset.batch_data(
    input_ids_train, attention_masks_train, slot_labels_train, batch_size=32)
val_dataloader = dataset.batch_data(
    input_ids_valid, attention_masks_valid, slot_labels_valid, batch_size=32)

# Create slots for each word in the text
"""
TODO: Implement the encode_token_labels function to mark subsequent subwords with a different placeholder (e.g., X).
for word, word_label in zip(text_sequence.split(), word_labels.split()):
    tokens = tokenizer.tokenize(word)
    encoded_labels.append(slot_map[word_label])  # First subword with correct label
    # Use 'X' or another placeholder for subsequent subwords
    encoded_labels.extend([slot_map.get('X', 0)] * (len(tokens) - 1))
"""


print("########################################### first lines_train:")
print(lines_train[:5])
print("########################################### df_train:")
print(df_train)
print("########################################### encoded_train_input_ids:")
print(encoded_train["input_ids"])
print("########################################### encoded_train_attention_masks:")
print(encoded_train["attention_masks"])
print("########################################### slot_map:")
print(slot_map)
print("########################################### slot_train:")
print(slot_train)
print("########################################### input_ids_train:")
print(input_ids_train)
print("########################################### attention_masks_train:")
print(attention_masks_train)
print("########################################### slot_labels_train:")
print(slot_labels_train)



# Define model and optimizer
config = BertConfig.from_pretrained("bert-base-cased")
config.return_dict = False
model = model.modeling_bert.JointIntentAndSlotFillingModel(slot_num_labels=len(slot_map))
optimizer = AdamW(model.parameters(), lr=3e-5)

# Define loss functions for slot and intent
# TODO Set ignore_index in CrossEntropyLoss to the [PAD] index to avoid penalizing padding in the loss calculation.
# pad_token_label_id = slot_map["[PAD]"]
# slot_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_label_id)
slot_loss_fn = nn.CrossEntropyLoss()
# intent_loss_fn = nn.CrossEntropyLoss()

# Define the Trainer
trainer = utils.trainer.Trainer(
    args=None,
    config=config,
    model=model,
    optimizer=optimizer,
    slot_loss_fn=slot_loss_fn,
    epochs=4,
    tokenizer=tokenizer,
    train_dataset=train_dataloader,
    val_dataset=val_dataloader,
    test_dataset=None
)

# Train the model
trainer.train()

# Save the model
#torch.save(model.state_dict(), "model.pth")
print("Model saved successfully.")


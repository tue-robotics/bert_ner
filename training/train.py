#!/usr/bin/env python3
"""
Train the NER slot tagger and save its weights as model.pth.

See docs/training.md for how to get training data and use the result.

    python3 training/train.py --train data/train.txt --valid data/valid.txt
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizer

# Use the model class (and slot vocabulary) the ROS package loads at runtime,
# so trained weights always match what inference expects.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ner_model.ner_model import JointIntentAndSlotFillingModel, get_device  # noqa: E402

import dataset  # noqa: E402
import training_loop  # noqa: E402

BASE_MODEL = "bert-base-cased"


def load_split(path, data, tokenizer, slot_map, max_length, batch_size):
    lines = Path(path).read_text("utf-8").strip().splitlines()
    df = pd.DataFrame([data.parse_line(line) for line in lines])
    encoded = data.encode_dataset(tokenizer, df["words"], max_length)
    labels = data.encode_token_labels(
        df["words"], df["word_labels"], tokenizer, slot_map, max_length
    )
    print("Loaded {} examples from {}".format(len(df), path))
    return data.batch_data(
        encoded["input_ids"], encoded["attention_masks"], torch.tensor(labels), batch_size
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--train", required=True, help="training file (word:label ... <=> action)")
    parser.add_argument("--valid", help="validation file, same format")
    parser.add_argument("--output", default="model.pth", help="where to save the weights")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=45)
    args = parser.parse_args()

    device = get_device()
    print("Training on {}".format(device))

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    slot_map = JointIntentAndSlotFillingModel.slot_map
    data = dataset.Dataset()

    train_data = load_split(args.train, data, tokenizer, slot_map, args.max_length, args.batch_size)
    valid_data = None
    if args.valid:
        valid_data = load_split(args.valid, data, tokenizer, slot_map, args.max_length, args.batch_size)

    config = BertConfig.from_pretrained(BASE_MODEL)
    config.return_dict = False
    model = JointIntentAndSlotFillingModel(slot_num_labels=len(slot_map), model_name=BASE_MODEL)
    model.to(device)

    model_trainer = training_loop.Trainer(
        args=None,
        config=config,
        model=model,
        optimizer=AdamW(model.parameters(), lr=args.lr),
        slot_loss_fn=nn.CrossEntropyLoss(),
        epochs=args.epochs,
        tokenizer=tokenizer,
        train_dataset=train_data,
        val_dataset=valid_data,
        device=device,
    )
    model_trainer.train(save_path=args.output)


if __name__ == "__main__":
    main()

from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Dataset():
    def __init__(self, max_length: int = 45, batch_size: int = 32):

        self.max_length = max_length
        self.batch_size = batch_size

    def parse_line(self, line):
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
    def encode_dataset(self, tokenizer, text_sequences, max_length):
        inputs = tokenizer.batch_encode_plus(
            text_sequences,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
            # TODO: return_offsets_mapping=True  # Add this line since it helps track token positions relative to the original words
        )
        return {"input_ids": inputs["input_ids"], "attention_masks": inputs["attention_mask"]}

    def encode_token_labels(self, text_sequences, slot_names, tokenizer, slot_map,
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
    
    def batch_data(self, input_ids, attention_masks, labels, batch_size):
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
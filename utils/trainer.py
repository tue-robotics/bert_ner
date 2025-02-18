import torch

#logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, config, model, optimizer, slot_loss_fn, epochs, tokenizer, train_dataset=None, val_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.epochs = epochs
        self.slot_loss_fn = slot_loss_fn
        self.tokenizer = tokenizer

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        #elf.pad_token_label_id = args.ignore_index

        # GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"we still have :D {self.device}")
        model.to(self.device)
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch in self.train_dataset:
                # Unpack batch
                input_ids, attention_mask, slot_labels = [b for b in batch]
                input_ids = input_ids.type(torch.LongTensor).to(self.device)
                attention_mask = attention_mask.type(torch.LongTensor).to(self.device)
                slot_labels = slot_labels.type(torch.LongTensor).to(self.device)
                slot_logits = self.model(input_ids, attention_mask=attention_mask)
                print("########################################### Forward_pass:")

                # Compute losses
                slot_loss = self.slot_loss_fn(
                    slot_logits.view(-1,
                                    slot_logits.shape[-1]), slot_labels.view(-1)
                )
                loss = slot_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(self.train_dataset)}")
        
        torch.save(self.model.state_dict(), "model.pth")

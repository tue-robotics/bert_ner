import torch
import time
from tqdm import tqdm

#logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, config, model, optimizer, slot_loss_fn, epochs, tokenizer, train_dataset=None, val_dataset=None, test_dataset=None, device=None):
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

        # Use provided device or fallback to CUDA/CPU detection
        if device is not None:
            self.device = device
            print(f"Using provided device: {self.device}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using fallback device detection: {self.device}")
        
        # Note: model should already be moved to device before passing to trainer
        
    def train(self):
        """
        Train the model with progress bars and detailed monitoring
        """
        print(f"Starting training for {self.epochs} epochs")
        print(f"Training batches: {len(self.train_dataset)}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        # Training history for monitoring
        training_history = []
        start_time = time.time()
        
        # Epoch progress bar
        epoch_pbar = tqdm(range(self.epochs), desc="Training Progress", position=0)
        
        for epoch in epoch_pbar:
            self.model.train()
            epoch_loss = 0
            batch_losses = []
            epoch_start = time.time()
            
            # Batch progress bar
            batch_pbar = tqdm(self.train_dataset, 
                            desc=f"Epoch {epoch+1}/{self.epochs}", 
                            position=1, 
                            leave=False)
            
            for batch_idx, batch in enumerate(batch_pbar):
                # Unpack batch
                input_ids, attention_mask, slot_labels = [b for b in batch]
                input_ids = input_ids.type(torch.LongTensor).to(self.device)
                attention_mask = attention_mask.type(torch.LongTensor).to(self.device)
                slot_labels = slot_labels.type(torch.LongTensor).to(self.device)
                
                # Forward pass
                slot_logits = self.model(input_ids, attention_mask=attention_mask)

                # Compute loss
                slot_loss = self.slot_loss_fn(
                    slot_logits.view(-1, slot_logits.shape[-1]), 
                    slot_labels.view(-1)
                )
                loss = slot_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_losses.append(batch_loss)
                
                # Update batch progress bar with latest metrics
                if batch_idx % 5 == 0: 
                    avg_batch_loss = sum(batch_losses[-10:]) / min(len(batch_losses), 10)  # Last 10 batches avg
                    batch_pbar.set_postfix({
                        'Loss': f'{batch_loss:.4f}',
                        'Avg': f'{avg_batch_loss:.4f}',
                        'Batch': f'{batch_idx+1}/{len(self.train_dataset)}'
                    })
            
            avg_epoch_loss = epoch_loss / len(self.train_dataset)
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            
            avg_epoch_time = elapsed_time / (epoch + 1)
            eta_seconds = avg_epoch_time * (self.epochs - epoch - 1)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'time': epoch_time
            })
            
            epoch_pbar.set_postfix({
                'Loss': f'{avg_epoch_loss:.4f}',
                'Time': f'{epoch_time:.1f}s',
                'ETA': eta_str
            })
            
            val_info = ""
            if self.val_dataset is not None:
                val_loss = self._validate()
                val_info = f" | Val Loss: {val_loss:.4f}"
            
            tqdm.write(f"Epoch {epoch+1:2d}/{self.epochs} | "
                      f"Train Loss: {avg_epoch_loss:.4f}{val_info} | "
                      f"Time: {epoch_time:.1f}s | ETA: {eta_str}")
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"Final Loss: {training_history[-1]['loss']:.4f}")
        print(f"Saving model to 'model.pth'...")
        
        torch.save(self.model.state_dict(), "model.pth")
        print("Model saved successfully!")
        
        return training_history
    
    def _validate(self):
        """
        Run validation and return average loss
        """
        if self.val_dataset is None:
            return None
            
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_dataset:
                input_ids, attention_mask, slot_labels = [b for b in batch]
                input_ids = input_ids.type(torch.LongTensor).to(self.device)
                attention_mask = attention_mask.type(torch.LongTensor).to(self.device)
                slot_labels = slot_labels.type(torch.LongTensor).to(self.device)
                
                slot_logits = self.model(input_ids, attention_mask=attention_mask)
                slot_loss = self.slot_loss_fn(
                    slot_logits.view(-1, slot_logits.shape[-1]), 
                    slot_labels.view(-1)
                )
                total_val_loss += slot_loss.item()
        
        return total_val_loss / len(self.val_dataset)

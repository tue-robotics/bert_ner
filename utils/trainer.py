import torch
import time
from tqdm import tqdm
from torch.amp import autocast, GradScaler

#logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, config, model, optimizer, slot_loss_fn, epochs, tokenizer, train_dataset=None, val_dataset=None, test_dataset=None, device=None, use_fp16=False, verbose_training=True):
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
        self.use_fp16 = use_fp16
        self.verbose_training = verbose_training

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        #elf.pad_token_label_id = args.ignore_index

        if device is not None:
            self.device = device
            print(f"Using provided device: {self.device}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using fallback device detection: {self.device}")
        
        # Initialize FP16 components
        if self.use_fp16:
            if self.device.type == "cuda":
                self.scaler = GradScaler()
                print("FP16 training enabled with CUDA AMP")
            elif self.device.type == "mps":
                # MPS supports FP16 but doesn't need GradScaler
                self.scaler = None
                print("FP16 training enabled with MPS")
            else:
                print("Warning: FP16 requested but not supported on CPU. Using FP32")
                self.use_fp16 = False
                self.scaler = None
        else:
            self.scaler = None
                
    def train(self):
        """
        Train the model with optional detailed monitoring
        """
        if self.verbose_training:
            print(f"Starting training for {self.epochs} epochs")
            print(f"Training batches: {len(self.train_dataset)}")
            print(f"Device: {self.device}")
            print("-" * 60)
        
        training_history = [] if self.verbose_training else None
        start_time = time.time() if self.verbose_training else None
        
        # Conditional progress bars
        if self.verbose_training:
            epoch_iterator = tqdm(range(self.epochs), desc="Training Progress", position=0)
        else:
            epoch_iterator = range(self.epochs)
        
        for epoch in epoch_iterator:
            self.model.train()
            epoch_loss = 0
            batch_losses = [] if self.verbose_training else None
            epoch_start = time.time() if self.verbose_training else None
            
            # Conditional batch progress bar
            if self.verbose_training:
                batch_iterator = tqdm(self.train_dataset, 
                                desc=f"Epoch {epoch+1}/{self.epochs}", 
                                position=1, 
                                leave=False)
            else:
                batch_iterator = self.train_dataset
            
            for batch_idx, batch in enumerate(batch_iterator):
                # Unpack batch
                input_ids, attention_mask, slot_labels = [b for b in batch]
                input_ids = input_ids.type(torch.LongTensor).to(self.device)
                attention_mask = attention_mask.type(torch.LongTensor).to(self.device)
                slot_labels = slot_labels.type(torch.LongTensor).to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.use_fp16:
                    # FP16 forward pass with autocast
                    with autocast(device_type=self.device.type):
                        slot_logits = self.model(input_ids, attention_mask=attention_mask)
                        slot_loss = self.slot_loss_fn(
                            slot_logits.view(-1, slot_logits.shape[-1]), 
                            slot_labels.view(-1)
                        )
                        loss = slot_loss
                    
                    # FP16 backward pass with scaling
                    if self.scaler is not None:  # CUDA
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:  # MPS
                        loss.backward()
                        self.optimizer.step()
                else:
                    # Standard FP32 training
                    slot_logits = self.model(input_ids, attention_mask=attention_mask)
                    slot_loss = self.slot_loss_fn(
                        slot_logits.view(-1, slot_logits.shape[-1]), 
                        slot_labels.view(-1)
                    )
                    loss = slot_loss
                    
                    # Standard backward pass
                    loss.backward()
                    self.optimizer.step()

                # Track metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss
                
                if self.verbose_training:
                    batch_losses.append(batch_loss)
                    
                    # Update batch progress bar with latest metrics
                    if batch_idx % 5 == 0: 
                        avg_batch_loss = sum(batch_losses[-10:]) / min(len(batch_losses), 10)  # Last 10 batches avg
                        batch_iterator.set_postfix({
                            'Loss': f'{batch_loss:.4f}',
                            'Avg': f'{avg_batch_loss:.4f}',
                            'Batch': f'{batch_idx+1}/{len(self.train_dataset)}'
                        })
            
            avg_epoch_loss = epoch_loss / len(self.train_dataset)
            
            if self.verbose_training:
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
                
                epoch_iterator.set_postfix({
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
            else:
                # Minimal output for non-verbose mode
                if self.val_dataset is not None:
                    val_loss = self._validate()
                    print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_epoch_loss:.4f}")
        
        if self.verbose_training:
            total_time = time.time() - start_time
            print("\n" + "="*60)
            print(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
            print(f"Final Loss: {training_history[-1]['loss']:.4f}")
            print(f"Saving model to 'model.pth'...")
        else:
            print("Training completed. Saving model...")
        
        torch.save(self.model.state_dict(), "model.pth")
        
        if self.verbose_training:
            print("Model saved successfully!")
        else:
            print("Model saved.")
        
        return training_history if self.verbose_training else None
    
    def _validate(self):
        
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
                
                if self.use_fp16:
                    # FP16 validation
                    with autocast(device_type=self.device.type):
                        slot_logits = self.model(input_ids, attention_mask=attention_mask)
                        slot_loss = self.slot_loss_fn(
                            slot_logits.view(-1, slot_logits.shape[-1]), 
                            slot_labels.view(-1)
                        )
                else:
                    # Standard FP32 validation
                    slot_logits = self.model(input_ids, attention_mask=attention_mask)
                    slot_loss = self.slot_loss_fn(
                        slot_logits.view(-1, slot_logits.shape[-1]), 
                        slot_labels.view(-1)
                    )
                
                total_val_loss += slot_loss.item()
        
        return total_val_loss / len(self.val_dataset)

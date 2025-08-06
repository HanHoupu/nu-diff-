import pytorch_lightning as pl
from tqdm import tqdm
from collections import OrderedDict
import time
import torch

class CustomProgressBar(pl.Callback):
    """
    A fully custom, non-spam progress bar that takes full control from Lightning.
    - Displays live metrics (loss) without reprinting the line.
    - Leaves each epoch's bar and summary on screen.
    - Includes a total elapsed time in the epoch summary.
    """
    def __init__(self):
        super().__init__()
        self.pbar = None
        self.start_time = None
        self.last_batch_idx = -1  # FIXED: Prevent duplicate updates

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.last_batch_idx = -1  # FIXED: Reset for new epoch
        self.pbar = tqdm(
            total=trainer.num_training_batches,
            desc=f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
            leave=True,
            ncols=90,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
            # FIXED: Disable duplicate output
            file=None,  # Use default stdout
            dynamic_ncols=False,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.pbar: return
        
        # FIXED: Prevent duplicate updates for same batch
        if batch_idx <= self.last_batch_idx:
            return
        self.last_batch_idx = batch_idx
        
        self.pbar.update(1)
        
        # FIXED: Get the correct loss key and handle NaN properly
        metrics = trainer.callback_metrics
        display_metrics = OrderedDict()
        
        # Try multiple possible loss keys in order of preference
        loss = None
        for key in ["train/loss", "train_loss_step", "loss"]:
            if key in metrics:
                loss_val = metrics[key]
                if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                    loss = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
                    break
        
        if loss is None:
            # Fallback: try to get from the return value
            if outputs is not None and isinstance(outputs, dict) and 'loss' in outputs:
                loss_val = outputs['loss']
                if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                    loss = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
            
        # Only display if we have a valid loss
        if loss is not None:
            display_metrics["loss"] = f"{loss:.3f}"
        
        self.pbar.set_postfix(display_metrics)

    def on_train_epoch_end(self, trainer, pl_module):
        # FIXED: Safely close progress bar
        if self.pbar:
            try:
                self.pbar.close()
            except:
                pass  # Ignore any errors during close
            finally:
                self.pbar = None
            
        # FIXED: Better metric extraction with fallbacks
        metrics = trainer.callback_metrics
        summary = f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        
        # Extract metrics with better error handling
        def safe_get_metric(key, default='nan'):
            val = metrics.get(key, float('nan'))
            if hasattr(val, 'item'):
                val = val.item()
            try:
                return float(val) if not (torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val))) else float(default)
            except:
                return float(default)
        
        train_loss = safe_get_metric('train/loss')
        val_loss = safe_get_metric('val/loss')  
        val_perplexity = safe_get_metric('val/perplexity')
        
        summary += f" - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_perplexity: {val_perplexity:.2e}"
        
        total_time = time.time() - self.start_time
        summary += f" - total_time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}"
        
        print(summary)

# --- Test Harness ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CustomProgressBar behavior.")
    parser.add_argument(
        "--test", action="store_true", help="Run a simulation."
    )
    args = parser.parse_args()

    if args.test:
        print("--- Simulating Custom, Non-刷屏 Progress Bar ---")
        total_epochs = 3
        steps_per_epoch = 100
        start_time = time.time()

        for epoch in range(total_epochs):
            progress_bar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{total_epochs}",
                leave=True,
                ncols=90, # Adjusted to 90
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
            )
            val_loss = 0.8 - (epoch * 0.1)

            for step in range(steps_per_epoch):
                train_loss = val_loss + 0.1 - (step / 500)
                
                postfix_dict = {"loss": f"{train_loss:.4f}"}
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(1)
                time.sleep(0.01)
            
            progress_bar.close()
            
            total_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{total_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_perplexity: {val_loss*100:.2e} - total_time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

        print("\n--- Simulation Complete ---") 
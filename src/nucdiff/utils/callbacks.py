import pytorch_lightning as pl
from tqdm import tqdm
from collections import OrderedDict
import time

class CustomProgressBar(pl.Callback):
    """
    A fully custom, non-刷屏 progress bar that takes full control from Lightning.
    - Displays live metrics (loss, R²) without reprinting the line.
    - Leaves each epoch's bar and summary on screen.
    - Includes a total elapsed time in the epoch summary.
    """
    def __init__(self):
        super().__init__()
        self.pbar = None
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.pbar = tqdm(
            total=trainer.num_training_batches,
            desc=f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
            leave=True,
            ncols=90,  # Adjusted to 90
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.pbar: return
        
        self.pbar.update(1)
        
        metrics = trainer.callback_metrics
        display_metrics = OrderedDict()
        
        # Use .get() to avoid errors if a metric is not yet available
        loss = metrics.get("train_loss_step", float('nan'))
        
        display_metrics["loss"] = f"{loss:.4f}"
        
        self.pbar.set_postfix(display_metrics)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.pbar:
            self.pbar.close()
            self.pbar = None
            
        metrics = trainer.callback_metrics
        summary = f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        
        train_loss = metrics.get('train_loss', float('nan'))
        val_loss = metrics.get('val_loss', float('nan'))
        val_perplexity = metrics.get('val_perplexity', float('nan'))
        
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
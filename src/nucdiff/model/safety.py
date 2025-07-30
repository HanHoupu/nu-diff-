# file: train_utils/safety.py
import torch
from torch import Tensor
from typing import Iterable, Optional

class SafetyCallback:
    """
    Training safety net:
      • NaN / Inf detection (loss, grad, param)
      • Gradient clipping (clip-norm or clip-value)
      • Custom hooks (e.g. log to TensorBoard / WandB)
    """

    def __init__(
        self,
        clip_norm: float = 1.0,        # ← Adjustable ①: L2-norm upper limit; set 0 to disable
        clip_value: Optional[float] = None,  # ← Adjustable ②: clip by absolute value; None to disable
        raise_on_nan: bool = True      # ← Adjustable ③: whether to raise exception on NaN/Inf detection
    ):
        self.clip_norm   = clip_norm
        self.clip_value  = clip_value
        self.raise_nan   = raise_on_nan

    # ---------- Called in each step ----------
    def __call__(self, loss: Tensor, model: torch.nn.Module):
        self._check_finite(loss, "loss")

        # Backward propagation
        loss.backward()

        # --- Gradient clipping ---
        if self.clip_norm and self.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)

        # --- Check gradients again ---
        for p in model.parameters():
            if p.grad is not None:
                self._check_finite(p.grad, "gradients")

    # ---------- Utility functions ----------
    def _check_finite(self, tensor: Tensor, tag: str):
        if not torch.isfinite(tensor).all():
            msg = f"[SafetyCallback] Detected non-finite {tag}."
            if self.raise_nan:
                raise RuntimeError(msg)
            else:
                print(msg)

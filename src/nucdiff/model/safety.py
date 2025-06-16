# file: train_utils/safety.py
import torch
from torch import Tensor
from typing import Iterable, Optional

class SafetyCallback:
    """
    训练安全网：
      • NaN / Inf 检测（loss、grad、param）
      • 梯度裁剪（clip-norm or clip-value）
      • 自定义 hooks（例如 log 到 TensorBoard / WandB）
    """

    def __init__(
        self,
        clip_norm: float = 1.0,        # ← 可调 ①：L2-norm 上限；设 0 则关闭
        clip_value: Optional[float] = None,  # ← 可调 ②：按绝对值裁剪；None 则关闭
        raise_on_nan: bool = True      # ← 可调 ③：发现 NaN/Inf 是否抛异常
    ):
        self.clip_norm   = clip_norm
        self.clip_value  = clip_value
        self.raise_nan   = raise_on_nan

    # ---------- 在每个 step 里调用 ----------
    def __call__(self, loss: Tensor, model: torch.nn.Module):
        self._check_finite(loss, "loss")

        # 反向传播
        loss.backward()

        # --- 梯度裁剪 ---
        if self.clip_norm and self.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)

        # --- 再次检查梯度是否正常 ---
        for p in model.parameters():
            if p.grad is not None:
                self._check_finite(p.grad, "gradients")

    # ---------- 工具函数 ----------
    def _check_finite(self, tensor: Tensor, tag: str):
        if not torch.isfinite(tensor).all():
            msg = f"[SafetyCallback] Detected non-finite {tag}."
            if self.raise_nan:
                raise RuntimeError(msg)
            else:
                print(msg)

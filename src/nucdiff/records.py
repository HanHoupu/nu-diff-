from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class QRecord:
    Z: int
    A: int
    element: str
    q_value_keV: Optional[float] = None
    mass_excess_keV: Optional[float] = None
    source: Optional[str] = None
    dataset_year: Optional[int] = None


@dataclass
class LRecord:
    level_id: int  # ✨ 唯一编号，方便 γ 过渡指向
    energy_keV: Optional[float]  # 能级能量
    jp: str  # 自旋-宇称 (e.g. "1/2+")
    half_life: str  # 寿命（原文字符串，先不换算）
    extras: Dict[str, Any] = field(default_factory=dict)  # 杂项键（%P, XREF …）
    source: Optional[str] = None
    dataset_year: Optional[int] = None


@dataclass
class GRecord:
    e_gamma_keV: Optional[float]  # γ 能量
    intensity: Optional[str]  # 相对强度 / 分支比（保原文）
    mul: Optional[str] = None  # 多极性 (M1, E2, M1(+E2)…)

    # —— 常用物理量直接数值化 ——
    width_ev: Optional[float] = None  # WIDTHG=…
    width_unc_ev: Optional[float] = None
    bm1w: Optional[float] = None  # B(M1)W
    be2w: Optional[float] = None  # B(E2)W

    # —— 能级配对索引 ——
    from_id: Optional[int] = None  # 起始能级 id
    to_id: Optional[int] = None  # 终止能级 id

    extras: Dict[str, Any] = field(default_factory=dict)  # 其余键 (FLAG, FL raw…)
    source: Optional[str] = None
    dataset_year: Optional[int] = None

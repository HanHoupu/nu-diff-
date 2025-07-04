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
    level_id: int  # unique id, convenient for gamma transition
    energy_keV: Optional[float]  # level energy
    jp: str  # spin-parity (e.g. "1/2+")
    half_life: str  # half-life (original string, not converted)
    extras: Dict[str, Any] = field(default_factory=dict)  # extra keys (%P, XREF ...)
    source: Optional[str] = None
    dataset_year: Optional[int] = None


@dataclass
class GRecord:
    e_gamma_keV: Optional[float]  # gamma energy
    intensity: Optional[str]  # relative intensity / branching ratio (keep original)
    mul: Optional[str] = None  # multipolarity (M1, E2, M1(+E2) ...)

    # -- commonly used physical quantities as numbers --
    width_ev: Optional[float] = None  # WIDTHG=...
    width_unc_ev: Optional[float] = None
    bm1w: Optional[float] = None  # B(M1)W
    be2w: Optional[float] = None  # B(E2)W

    # -- level pairing index --
    from_id: Optional[int] = None  # from level id
    to_id: Optional[int] = None  # to level id

    extras: Dict[str, Any] = field(default_factory=dict)  # other keys (FLAG, FL raw ...)
    source: Optional[str] = None
    dataset_year: Optional[int] = None

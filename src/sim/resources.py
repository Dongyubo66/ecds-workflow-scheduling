from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from .energy import DVFSPowerModel


@dataclass(frozen=True)
class GreenSegment:
    start: float
    green: float


@dataclass
class Machine:
    name: str
    speed: float
    f: float
    power: DVFSPowerModel

    site: str = "default"

    green_ratio: float = 0.0
    green_profile: List[GreenSegment] = field(default_factory=list)

    ci_green: float = 0.05
    ci_brown: float = 0.55

    def green_fraction_at(self, t: float) -> float:
        if not self.green_profile:
            return float(min(1.0, max(0.0, self.green_ratio)))

        prof = sorted(self.green_profile, key=lambda x: float(x.start))
        seg = prof[0]
        tt = float(t)
        for s in prof:
            if float(s.start) <= tt:
                seg = s
            else:
                break
        return float(min(1.0, max(0.0, float(seg.green))))

    def avg_green_fraction(self, start: float, finish: float) -> float:
        s = float(start)
        f = float(finish)
        if f <= s + 1e-12:
            return self.green_fraction_at(s)

        if not self.green_profile:
            return float(min(1.0, max(0.0, self.green_ratio)))

        prof = sorted(self.green_profile, key=lambda x: float(x.start))
        starts = [float(seg.start) for seg in prof]

        total = 0.0
        t = s
        while t < f - 1e-12:
            g = self.green_fraction_at(t)
            nxt = f
            for st in starts:
                if st > t + 1e-12:
                    nxt = min(nxt, st)
            dt = max(0.0, min(f, nxt) - t)
            total += g * dt
            t += dt

        return float(min(1.0, max(0.0, total / (f - s))))

    @property
    def efficiency(self) -> float:
        # speed per total power (dynamic + static)
        p = self.power.p_dynamic(self.f) + self.power.p_idle()
        return float(self.speed) / max(1e-9, float(p))

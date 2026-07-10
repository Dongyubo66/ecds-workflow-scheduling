from dataclasses import dataclass

@dataclass(frozen=True)
class DVFSPowerModel:
    a: float
    b: float

    def p_dynamic(self, f: float) -> float:
        ff = max(0.0, float(f))
        return float(self.a) * (ff ** 3)

    def p_idle(self) -> float:
        return float(self.b)

    def energy(self, f: float, exec_time: float, slot_time: float) -> float:
        t_exec = max(0.0, float(exec_time))
        t_slot = max(t_exec, float(slot_time))
        return self.p_dynamic(f) * t_exec + self.p_idle() * t_slot
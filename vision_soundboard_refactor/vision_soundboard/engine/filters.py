
import math, time

class OneEuroFilter:
    def __init__(self, freq=60.0, mincutoff=0.5, beta=0.005, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def set_params(self, mincutoff=None, beta=None, dcutoff=None):
        """Set filter parameters for OneEuroFilter."""
        if mincutoff is not None:
            self.mincutoff = float(mincutoff)
        if beta is not None:
            self.beta = float(beta)
        if dcutoff is not None:
            self.dcutoff = float(dcutoff)

    def set_dynamic_params(self, quality):
        """Dynamically adjust filter parameters based on frame quality."""
        # ตัวอย่าง logic: quality ต่ำ -> หนืดขึ้น, quality สูง -> ตอบสนองเร็วขึ้น
        self.mincutoff = max(0.3, 1.6 - 1.2 * quality)
        self.beta = 0.01 + 0.12 * (1.0 - quality)

    def reset(self, x=None):
        """Reset filter state. Optionally set initial value x."""
        self.x_prev = x
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff):
        te = 1.0 / max(1e-6, self.freq)
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if self.t_prev is None:
            self.t_prev = time.time() if t is None else t
            self.x_prev = x
            return x
        tnow = time.time() if t is None else t
        dt = max(1e-6, tnow - self.t_prev)
        self.freq = 1.0 / dt
        self.t_prev = tnow
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

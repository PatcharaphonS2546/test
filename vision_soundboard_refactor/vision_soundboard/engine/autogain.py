
from collections import deque

class AutoGain:
    def __init__(self, target=0.48, window_sec=2.5, ema_alpha=0.05):
        self.target = float(target)
        self.window_sec = float(window_sec)
        self.ema_alpha = float(ema_alpha)
        self.samples = deque()
        self.gain = 1.0
        self.frozen = False

    def reset(self):
        self.samples.clear()
        self.gain = 1.0

    def set_frozen(self, frozen: bool):
        self.frozen = bool(frozen)

    def update(self, abs_val, now_ts):
        self.samples.append((now_ts, float(abs_val)))
        cut = now_ts - self.window_sec
        while self.samples and self.samples[0][0] < cut:
            self.samples.popleft()
        if self.frozen or len(self.samples) < 8:
            return self.gain
        vals = [v for _, v in self.samples]
        vals.sort()
        idx = int(0.95 * (len(vals)-1))
        perc = max(1e-4, vals[idx])
        target_gain = self.target / perc
        self.gain = (1.0 - self.ema_alpha)*self.gain + self.ema_alpha*target_gain
        return self.gain

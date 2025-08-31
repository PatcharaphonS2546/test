
def clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def signed_gamma(u: float, gamma: float) -> float:
    v = u - 0.5
    s = 1.0 if v >= 0 else -1.0
    a = abs(v)
    a_g = a ** max(1e-6, gamma)
    return clamp01(0.5 + s * a_g)

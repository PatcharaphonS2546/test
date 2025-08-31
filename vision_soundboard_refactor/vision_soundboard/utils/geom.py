
from typing import Tuple

def inside_rect(nx: float, ny: float, rect: Tuple[float,float,float,float]) -> bool:
    x0, y0, x1, y1 = rect
    return (x0 <= nx <= x1) and (y0 <= ny <= y1)

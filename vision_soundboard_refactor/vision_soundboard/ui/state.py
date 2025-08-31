
from dataclasses import dataclass, field
from typing import List, Tuple
from ..utils.screen import safe_screen_size

@dataclass
class AppState:
    def set_param(self, name: str, value):
        """Set parameter from UI in real-time."""
        if hasattr(self, name):
            setattr(self, name, value)
    # เก็บ history ของ gaze point
    gaze_history: List[Tuple[float, float]] = field(default_factory=list)

    def add_gaze_history(self, x: float, y: float, maxlen: int = 100):
        """Add gaze point to history, keep only last maxlen points."""
        self.gaze_history.append((float(x), float(y)))
        if len(self.gaze_history) > maxlen:
            self.gaze_history = self.gaze_history[-maxlen:]
    def update_gaze(self, x: float, y: float, quality: float, threshold: float = 0.2):
        """Update gaze point only if quality is above threshold, otherwise smooth or ignore."""
        if quality >= threshold:
            self.gx = float(x)
            self.gy = float(y)
        else:
            # Smooth with previous value if quality ต่ำ
            self.gx = 0.8 * self.gx + 0.2 * float(x)
            self.gy = 0.8 * self.gy + 0.2 * float(y)
    # Compensation params
    k_yaw: float = 0.35
    c_yaw: float = 0.08
    k_pitch: float = 0.30
    c_pitch: float = 0.06

    # Bias / autogain / shaping
    bias_x: float = 0.0
    bias_y: float = 0.0
    autogain_freeze: bool = False
    autogain_reset: bool = False
    gamma_x: float = 1.15
    gamma_y: float = 1.15

    # Mode + flags
    saccade_aware: bool = True
    mode: str = "Webcam/MediaPipe"
    mirror: bool = False
    invert_x: bool = False
    invert_y: bool = False
    gain: float = 1.0
    gamma: float = 1.0
    deadzone: float = 0.02

    # Screen
    use_screen_override: bool = False
    screen_w: int = field(default_factory=lambda: safe_screen_size()[0])
    screen_h: int = field(default_factory=lambda: safe_screen_size()[1])
    deg_per_px: float = 0.0

    # Mouse
    mouse_enabled: bool = False

    # Calibration
    calib_overlay: bool = False
    # สามารถปรับจำนวนจุดและตำแหน่ง calibration ได้จากภายนอก
    targets: List[Tuple[float,float]] = field(default_factory=lambda: AppState.default_targets())

    @staticmethod
    def default_targets(rows: int = 4, cols: int = 4) -> List[Tuple[float, float]]:
        """Generate default calibration targets grid."""
        return [(x/(cols-1), y/(rows-1)) for y in range(rows) for x in range(cols)]
    idx: int = 0
    dwell_ms: int = 1000
    radius_norm: float = 0.02
    countdown_secs: int = 3
    countdown_active: bool = False
    countdown_end: float = 0.0
    # เก็บ quality ของแต่ละ calibration point
    calib_qualities: List[float] = field(default_factory=list)

    # Shared gaze + metrics
    gx: float = 0.5
    gy: float = 0.5
    ui_fps: float = 0.0
    ui_lat: float = 0.0

    # Soundboard (2x3) Thai
    soundboard_on: bool = False
    sound_labels: List[str] = field(default_factory=lambda: ["สวัสดี", "ใช่", "ไม่ใช่", "ขอบคุณ", "ช่วยด้วย", "ไปห้องน้ำ"])
    sound_files: List[str]  = field(default_factory=lambda: [
        "C:/sound/back.mp3",
        "C:/sound/head.mp3",
        "C:/sound/Hungry.mp3",
        "C:/sound/snack.mp3",
        "C:/sound/stoma.mp3",
        "C:/sound/Thirsty.mp3",
    ])
    sound_icons: List[str]  = field(default_factory=lambda: [
        "C:/icon/question.png",
        "C:/icon/pain.png",
        "C:/icon/emergency.png",
        "C:/icon/toilet.png",
        "C:/icon/temperature.png",
        "C:/icon/cutlery.png",
    ])
    sound_colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (255, 0, 0),    # ปุ่มที่ 1 สีแดง
        (0, 255, 0),    # ปุ่มที่ 2 สีเขียว
        (0, 0, 255),    # ปุ่มที่ 3 สีน้ำเงิน
        (255, 255, 0),  # ปุ่มที่ 4 สีเหลือง
        (255, 0, 255),  # ปุ่มที่ 5 สีม่วง
        (0, 255, 255),  # ปุ่มที่ 6 สีฟ้า
    ])
    sound_dwell_ms: int = 3000
    sound_cooldown_ms: int = 1500
    _sound_current_idx: int | None = None
    _sound_start: float = 0.0
    _sound_last_play: List[float] = field(default_factory=lambda: [-1.0]*6)

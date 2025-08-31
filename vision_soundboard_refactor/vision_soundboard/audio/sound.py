
import threading as _th

try:
    import simpleaudio as sa  # best for .wav
except Exception:  # pragma: no cover
    sa = None

try:
    from playsound import playsound as _playsound
except Exception:  # pragma: no cover
    _playsound = None

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None

def play_sound_file(path: str) -> bool:
    """Play an audio file non-blocking on the server machine.
    Returns True if some backend accepted the file.
    """
    try:
        if sa and path.lower().endswith(('.wav', '.wave')):
            wave_obj = sa.WaveObject.from_wave_file(path)
            wave_obj.play()
            return True
        elif path.lower().endswith('.mp3') and pygame:
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                return True
            except Exception:
                pass
        elif _playsound:
            _th.Thread(target=_playsound, args=(path,), daemon=True).start()
            return True
    except Exception as e:  # pragma: no cover
        print("play_sound_file failed:", e)
    return False

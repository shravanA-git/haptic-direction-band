"""OS thread priority, so heavy background work (the camera/YOLO loop) cannot starve the name and
sound pipeline. macOS only: pthread QoS classes, which on Apple Silicon also decide the core type
(utility/background threads go to the efficiency cores, leaving the performance cores to whisper).
Everywhere else this does nothing and returns False. Never raises."""
from __future__ import annotations

import sys

QOS = {"interactive": 0x21, "initiated": 0x19, "default": 0x15, "utility": 0x11, "background": 0x09}
_lib = None


def set_thread_qos(level: str) -> bool:
    """Set the CALLING thread's QoS class: interactive | initiated | default | utility | background.
    Threads it creates afterwards inherit it. True if the OS accepted it."""
    global _lib
    if sys.platform != "darwin" or level not in QOS:
        return False
    try:
        import ctypes
        if _lib is None:
            _lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        return _lib.pthread_set_qos_class_self_np(ctypes.c_uint(QOS[level]), ctypes.c_int(0)) == 0
    except Exception:
        return False

"""TensorFlow device and reproducibility setup."""

import random

import numpy as np
import tensorflow as tf


def set_reproducible(seed: int | None) -> None:
    """Reset TensorFlow and NumPy state so repeated runs are comparable."""
    tf.keras.backend.clear_session()
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def configure_tensorflow_device(device: str = "auto") -> None:
    """Select CPU/GPU execution without changing model behavior."""
    device = device.lower().strip()
    if device not in {"auto", "gpu", "cpu"}:
        raise ValueError("device parametresi 'auto', 'gpu' veya 'cpu' olmalidir.")

    available_gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not available_gpus:
        raise RuntimeError("GPU istendi ancak TensorFlow tarafindan GPU bulunamadi.")

    try:
        if device == "cpu":
            tf.config.set_visible_devices([], "GPU")
            print("[INFO] GPU devre disi birakildi, CPU uzerinden calisiyor.")
            return
        if available_gpus:
            for gpu in available_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(available_gpus, "GPU")
            print(f"[INFO] GPU uzerinden calisacak: {[gpu.name for gpu in available_gpus]}")
        else:
            print("[INFO] GPU bulunamadi, CPU uzerinden calisiyor.")
    except Exception as exc:
        print(f"[WARN] TensorFlow cihaz ayarlari yapilamadi: {exc}")

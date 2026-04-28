"""
models/__init__.py
------------------
Factory para instanciar modelos por nombre.
"""

from .spectnet      import SpectNet
from .eegnet        import EEGNet
from .shallowconvnet import ShallowConvNet


def get_model(name: str, **kwargs):
    """
    Retorna una instancia del modelo indicado.

    Parámetros
    ----------
    name   : 'spectnet' | 'eegnet' | 'shallowconvnet'
    kwargs : parámetros del constructor (n_channels, n_freq, n_time, n_classes, ...)

    Ejemplo
    -------
    model = get_model('spectnet', n_channels=3, n_freq=22, n_time=128, n_classes=2)
    """
    name = name.lower().strip()
    registry = {
        "spectnet":       SpectNet,
        "eegnet":         EEGNet,
        "shallowconvnet": ShallowConvNet,
    }
    if name not in registry:
        raise ValueError(
            f"Modelo '{name}' no reconocido. "
            f"Opciones: {list(registry.keys())}"
        )
    return registry[name](**kwargs)


__all__ = ["SpectNet", "EEGNet", "ShallowConvNet", "get_model"]

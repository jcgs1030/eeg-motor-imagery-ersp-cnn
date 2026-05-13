"""
models/__init__.py
------------------
Factory for instantiating models by name.
"""

from .spectnet      import SpectNet
from .eegnet        import EEGNet
from .shallowconvnet import ShallowConvNet


def get_model(name: str, **kwargs):
    """
    Return an instance of the requested model.

    Parameters
    ----------
    name   : 'spectnet' | 'eegnet' | 'shallowconvnet'
    kwargs : constructor parameters (n_channels, n_freq, n_time, n_classes, ...)

    Example
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
            f"Unknown model '{name}'. "
            f"Options: {list(registry.keys())}"
        )
    return registry[name](**kwargs)


__all__ = ["SpectNet", "EEGNet", "ShallowConvNet", "get_model"]

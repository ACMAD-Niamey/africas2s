_METHODS = {}
_METRICS = {}
_STRATEGIES = {}
_CALIBRATORS = {}

def register_method(name):
    def decorator(cls):
        _METHODS[name] = cls
        return cls
    return decorator

def register_calibrator(name):
    """Register a calibrate-family method (probabilistic post-processing that
    emits tercile probabilities with no resolution change), distinct from the
    downscale-family methods registered with ``register_method``."""
    def decorator(fn):
        _CALIBRATORS[name] = fn
        return fn
    return decorator

def register_metric(name, *, aliases=()):
    def decorator(cls):
        _METRICS[name] = cls
        for alias in aliases:
            _METRICS[alias] = cls
        return cls
    return decorator

def register_strategy(name):
    def decorator(cls):
        _STRATEGIES[name] = cls
        return cls
    return decorator

def get_method(name):
    if name not in _METHODS:
        raise KeyError(f"Unknown method: {name}")
    return _METHODS[name]

def get_metric(name):
    if name not in _METRICS:
        raise KeyError(f"Unknown metric: {name}")
    return _METRICS[name]

def get_strategy(name):
    if name not in _STRATEGIES:
        raise KeyError(f"Unknown strategy: {name}")
    return _STRATEGIES[name]

def get_calibrator(name):
    if name not in _CALIBRATORS:
        raise KeyError(
            f"Unknown calibrate method: {name!r}. Available: {sorted(_CALIBRATORS)}."
        )
    return _CALIBRATORS[name]

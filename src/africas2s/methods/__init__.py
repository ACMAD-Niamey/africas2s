from . import (bcsd, cca, cccma_calibration, chelsa, climatology, delta, dqm, qm,
               rank_analog, smoothed_regression)

try:
    from . import corrdiff
except ImportError:
    pass

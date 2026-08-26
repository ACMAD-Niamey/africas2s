from . import uniform
from . import drop_worst
from . import skill_weighted
from . import bma

# Note: `regime_dependent` (Vigaud et al. 2017) is intentionally not yet
# implemented — its design depends on an ENSO-regime label per year, which
# in turn depends on the §11.2 teleconnection-indices Rosetta product
# (issue #33). Once that lands, regime_dependent can be added with a BYO
# regime-labels API.

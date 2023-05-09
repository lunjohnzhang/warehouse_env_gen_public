"""pyribs-compliant emitters."""
import gin
import ribs

from env_search.emitters.map_elites_baseline_emitter import MapElitesBaselineEmitter
from env_search.emitters.random_emitter import RandomEmitter

__all__ = [
    "GaussianEmitter",
    "ImprovementEmitter",
    "MapElitesBaselineEmitter",
    "RandomEmitter",
]


@gin.configurable
class GaussianEmitter(ribs.emitters.GaussianEmitter):
    """gin-configurable version of pyribs GaussianEmitter."""


@gin.configurable
class ImprovementEmitter(ribs.emitters.ImprovementEmitter):
    """gin-configurable version of pyribs ImprovementEmitter."""

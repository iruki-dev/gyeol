"""Discovery on the residual r(t) and phonation features (M7).

* :mod:`~gyeol.discover.sae` — TopK sparse autoencoder (+ AuxK dead-latent revival);
* :mod:`~gyeol.discover.match` — match SAE features to labels and v0.1 DSP features; novel candidates;
* :mod:`~gyeol.discover.directions` — conditional directions from paired on/off data (f0/loudness regressed out);
* :mod:`~gyeol.discover.transfer` — held-out singer / pitch-range / language transfer tests, promotion rule, registry;
* :mod:`~gyeol.discover.residual` — residual-energy monitoring (rising residual = missing coverage).

Nothing here produces coaching output directly: a discovered feature reaches
the coach only after promotion and training as an attribute head.
"""

from .directions import ConditionalDirections, Direction, DirectionConfig, PairedFrames, concat, fit_conditional_directions, pair_frames, pair_from_example
from .match import FeatureMatch, MatchTable, auroc, match_features
from .residual import ResidualMonitor, ResidualReport, representation_residual_energy, residual_energy
from .sae import FeatureStats, SAEConfig, SAEFit, TopKSAE, feature_stats, train_sae
from .transfer import (
    PromotedAttribute,
    PromotionDecision,
    PromotionError,
    PromotionRegistry,
    TransferConfig,
    TransferResult,
    evaluate_promotion,
)

__all__ = [
    "ConditionalDirections", "Direction", "DirectionConfig", "FeatureMatch", "FeatureStats", "MatchTable", "PairedFrames",
    "PromotedAttribute", "PromotionDecision", "PromotionError", "PromotionRegistry", "ResidualMonitor", "ResidualReport",
    "SAEConfig", "SAEFit", "TopKSAE", "TransferConfig", "TransferResult", "auroc", "concat", "evaluate_promotion",
    "feature_stats", "fit_conditional_directions", "match_features", "pair_frames", "pair_from_example",
    "representation_residual_energy", "residual_energy", "train_sae",
]

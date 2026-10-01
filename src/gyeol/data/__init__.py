"""Dataset adapters, manifests, splits, augmentation and paired loading (M2)."""

from .adapters import (
    PHONATION_LABELS,
    AIHubFieldMap,
    ScanReport,
    inspect_json_keys,
    scan_aihub,
    scan_gtsinger,
    scan_own,
    scan_vocalset,
)
from .manifest import Dataset, Manifest, ManifestItem, open_manifest
from .paired import PairedExample, PairedLoader
from .splits import pairs, singer_split

__all__ = [
    "AIHubFieldMap", "Dataset", "Manifest", "ManifestItem", "PHONATION_LABELS", "PairedExample", "PairedLoader",
    "ScanReport", "inspect_json_keys", "open_manifest", "pairs", "scan_aihub", "scan_gtsinger", "scan_own", "scan_vocalset",
    "singer_split",
]

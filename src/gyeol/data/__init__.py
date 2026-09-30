"""Dataset adapters and manifests with license tags, splits, augmentation and paired loading (M0 gate, M2)."""

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
from .manifest import LicensedDataset, Manifest, ManifestItem, open_manifest
from .paired import PairedExample, PairedLoader
from .splits import pairs, singer_split

__all__ = [
    "AIHubFieldMap", "LicensedDataset", "Manifest", "ManifestItem", "PHONATION_LABELS", "PairedExample", "PairedLoader",
    "ScanReport", "inspect_json_keys", "open_manifest", "pairs", "scan_aihub", "scan_gtsinger", "scan_own", "scan_vocalset",
    "singer_split",
]

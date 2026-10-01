"""License tags and their enforcement.

Every dataset, checkpoint and third-party model carries a :class:`LicenseTag`.
Training and inference run under a :class:`Profile`; under ``COMMERCIAL`` the
data and checkpoint loaders call :func:`require_allowed`, which *refuses*
anything that is not commercially usable.

The registry below records the tags given in the gyeol v2 design brief.  They
have **not** been re-verified against upstream by this code (``verified`` is
False); verify each one before relying on it and flip the flag in a reviewed
change.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum


class LicenseTag(str, Enum):
    COMMERCIAL_OK = "commercial_ok"
    #: usable commercially only under stated conditions (e.g. AI Hub)
    COMMERCIAL_OK_CONDITIONAL = "commercial_ok_conditional"
    NONCOMMERCIAL = "noncommercial"
    COPYLEFT = "copyleft"
    UNKNOWN = "unknown"


class Profile(str, Enum):
    COMMERCIAL = "commercial"
    RESEARCH = "research"


class AssetKind(str, Enum):
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    MODEL = "model"  # third-party code / architecture


class LicenseError(PermissionError):
    """Raised when an asset is not allowed under the active profile."""


@dataclass(frozen=True)
class LicensedAsset:
    name: str
    kind: AssetKind
    tag: LicenseTag
    license: str  # SPDX-ish string, e.g. "CC-BY-4.0"
    source: str = ""
    conditions: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    #: may the code/weights be copied into this repository?
    vendor_allowed: bool = True
    #: tag re-checked against upstream by a maintainer
    verified: bool = False
    url: str | None = None  # download location used by ``gyeol fetch``; None = manual
    sha256: str | None = None


_AIHUB_CONDITIONS = (
    "Trained models may be used commercially with attribution to AI Hub (NIA).",
    "Raw data must never be redistributed.",
    "Raw data must be processed domestically (Republic of Korea).",
    "Each user must download the data individually from aihub.or.kr under their own account.",
)


def _a(name, kind, tag, lic, source="", conditions=(), caveats=(), vendor_allowed=True, url=None, verified=False) -> LicensedAsset:
    return LicensedAsset(name, kind, tag, lic, source, tuple(conditions), tuple(caveats), vendor_allowed, verified, url)


D, C, M = AssetKind.DATASET, AssetKind.CHECKPOINT, AssetKind.MODEL
OK, OKC, NC, CL = LicenseTag.COMMERCIAL_OK, LicenseTag.COMMERCIAL_OK_CONDITIONAL, LicenseTag.NONCOMMERCIAL, LicenseTag.COPYLEFT

REGISTRY: dict[str, LicensedAsset] = {
    a.name: a
    for a in [
        # datasets
        _a("vocalset", D, OK, "CC-BY-4.0", "VocalSet (Wilkins et al., ISMIR 2018)"),
        _a("gtsinger", D, NC, "CC-BY-NC-SA-4.0", "GTSinger"),
        _a("popbutfy", D, NC, "CC-BY-NC-SA", "PopBuTFy"),
        _a("csd", D, NC, "CC-BY-NC-SA-4.0", "https://github.com/equal-singer/CSD"),
        _a("opencpop", D, NC, "CC-BY-NC", "Opencpop"),
        _a("vocalcoachbench", D, NC, "mixed per source: Smule Research Data License (DAMP), CC-BY-NC-SA-4.0, CC-BY(-SA)-4.0",
           "VocalCoachBench (arXiv:2609.04241)",
           caveats=("Audio keeps each source corpus's license; several are research-only or non-commercial.",
                    "The annotation license is not stated in the paper; verify it at download.",
                    "English singing, expert notes translated to English: evaluation only.")),
        _a("aihub_473_guide_vocal", D, OKC, "AI Hub terms", "https://aihub.or.kr (다음색 가이드보컬 #473)", _AIHUB_CONDITIONS),
        _a("aihub_465_multi_singer", D, OKC, "AI Hub terms", "https://aihub.or.kr (다화자 가창 #465)", _AIHUB_CONDITIONS),
        # checkpoints / models
        _a("bigvgan_v2", C, OK, "MIT", "NVIDIA BigVGAN"),
        _a("vocos", C, OK, "MIT", "Vocos"),
        _a("dac", C, OK, "MIT", "Descript Audio Codec"),
        _a("hubert_fairseq", C, OK, "MIT", "fairseq HuBERT"),
        _a("contentvec", C, OK, "MIT", "ContentVec"),
        _a("rmvpe", C, OK, "Apache-2.0", "RMVPE"),
        _a("fcpe", C, OK, "MIT", "torchfcpe"),
        _a("swiftf0", C, OK, "MIT", "SwiftF0"),
        _a("roformer_community", C, OK, "MIT", "community Mel/BS-RoFormer weights",
           caveats=("Training-data provenance of the community weights is unclear.",)),
        _a("bs_roformer_viperx_ep317", C, OK, "MIT (as listed for community RoFormer weights)",
           "viperx BS-RoFormer vocal model (model_bs_roformer_ep_317_sdr_12.9755; UVR public model repository); "
           "architecture: gyeol.frontend.roformer.BSRoFormer with VIPERX_EP317",
           caveats=("Training-data provenance of the community weights is unclear.",
                    "The model repository states no separate weights license; verify it before a commercial release.",
                    "Checksum not pinned yet: record the SHA-256 of your first verified download in the registry."),
           url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
        _a("openvpi_nsf_hifigan", C, NC, "CC-BY-NC-SA-4.0", "openvpi vocoders (pretrained weights)"),
        _a("openvpi_pc_nsf_hifigan", C, NC, "CC-BY-NC-SA-4.0", "openvpi vocoders (pretrained weights)"),
        _a("gyeol_synthetic", D, OK, "generated (no recordings, no people)", "gyeol.synth / gyeol.eval.knob_recovery / gyeol.data.prepare",
           verified=True),
        _a("own_recordings", D, OK, "in-house (user consent)", "recordings made in the gyeol app",
           conditions=("Only recordings whose owner consents to the training purpose may be used (enforced by data.adapters.scan_own).",)),
        _a("so_vits_svc", M, CL, "AGPL-3.0", "so-vits-svc", vendor_allowed=False),
        _a("pesto", M, CL, "LGPL-3.0", "PESTO", vendor_allowed=False),
    ]
}


@dataclass
class LicenseDecision:
    allowed: bool
    reason: str
    notices: list[str] = field(default_factory=list)


def decide(asset: LicensedAsset, profile: Profile) -> LicenseDecision:
    notices = list(asset.caveats)
    if not asset.verified:
        notices.append(f"license tag of {asset.name!r} has not been re-verified upstream")
    if profile is Profile.COMMERCIAL:
        if asset.tag is LicenseTag.COMMERCIAL_OK:
            return LicenseDecision(True, "commercial_ok", notices)
        if asset.tag is LicenseTag.COMMERCIAL_OK_CONDITIONAL:
            return LicenseDecision(True, "commercial use permitted under conditions", list(asset.conditions) + notices)
        return LicenseDecision(False, f"{asset.name!r} is tagged {asset.tag.value} ({asset.license}); refused under the commercial profile", notices)
    # research profile: everything loads, with notices
    if asset.tag is LicenseTag.COPYLEFT:
        notices.append(f"{asset.name!r} is copyleft ({asset.license}); do not vendor or link it into distributed code")
    if asset.tag is LicenseTag.UNKNOWN:
        notices.append(f"{asset.name!r} has an unknown license; research use only")
    if asset.tag is LicenseTag.COMMERCIAL_OK_CONDITIONAL:
        notices.extend(asset.conditions)
    return LicenseDecision(True, "research profile", notices)


def require_allowed(asset: LicensedAsset, profile: Profile, *, announce: bool = True) -> LicenseDecision:
    """Raise :class:`LicenseError` if ``asset`` may not be used under ``profile``.

    Conditions (e.g. AI Hub terms) are printed to stderr when ``announce``.
    """
    d = decide(asset, Profile(profile))
    if not d.allowed:
        raise LicenseError(d.reason)
    if announce:
        for n in d.notices:
            if n in asset.conditions:
                print(f"[gyeol license] {asset.name}: {n}", file=sys.stderr)
            else:
                warnings.warn(f"[gyeol license] {n}", stacklevel=2)
    return d


def register(asset: LicensedAsset, *, replace: bool = False) -> None:
    """Add an asset (e.g. a new dataset or checkpoint) to the registry."""
    if asset.name in REGISTRY and not replace:
        raise ValueError(f"{asset.name!r} is already registered")
    REGISTRY[asset.name] = asset


def lookup(name: str) -> LicensedAsset:
    try:
        return REGISTRY[name]
    except KeyError:
        raise LicenseError(f"asset {name!r} is not in the license registry; register it with a LicenseTag first") from None


def unknown_asset(name: str, kind: AssetKind) -> LicensedAsset:
    return LicensedAsset(name, kind, LicenseTag.UNKNOWN, "unknown")

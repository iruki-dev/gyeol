"""Real-recording evaluation set (revision A6): manifest, split and ``gyeol eval realset``.

The user supplies the recordings; gyeol supplies the format and the metrics.

Manifest — ``manifest.jsonl`` in the set's folder, one JSON object per recording::

    {"id": "s01_t1", "audio": "audio/s01_t1.wav", "role": "user", "target": "song03",
     "singer": "s01", "session": "s01-2026-09-01", "condition": "mixture_phone",
     "lyrics": "사랑해 너를",                                    # optional
     "annotations": {                                            # all optional
        "f0": "ann/s01_t1.f0.csv",          # time_s,f0_hz per line (0 = unvoiced)
        "onsets": [0.21, 0.83, 1.40],       # sung note onsets (s) — or a .txt path, one per line
        "octave_relation": -1,              # user − target in octaves (or "transposition_semitones": n)
        "rhythm_ok": true,                  # the annotator judged the timing correct …
        "onset_deviation_ms": {"3": 80}     # … or per-note deviations (note index → ms, + = late)
     },
     "recording": {"device": "phone", "route": "bluetooth", "room": "living room",
                   "backing": "backing/song03.wav"}}   # backing track, if known (sing-along)

    {"id": "song03", "audio": "audio/song03_guide.wav", "role": "target", "condition": "clean"}

Conditions: ``mixture_phone``, ``mixture_karaoke`` (with accompaniment),
``separated`` (vocals separated beforehand), ``clean``, ``trimmed`` (hand-cut
clips that do not share the target's clock) and ``multi_singer`` (more than one
voice).  Unknown conditions are refused, not guessed.

Split — tuning / held-out by **singer** (every session of a singer stays on one
side, so the split is also session-disjoint); targets are shared.  The split is
written to ``split.json`` so later runs reuse it.

Metrics per condition:

* octave decisions made / withheld / wrong (against ``octave_relation``);
* false rhythm verdicts — a rhythm item at or above the verdict threshold
  where the annotation says the timing was right (and the rate per verdict);
* withheld rate — judgements withheld by failed premises, and how often each
  premise failed;
* agreement — raw pitch accuracy (|Δ| ≤ 50 cents on annotated voiced frames),
  octave-folded chroma accuracy, voicing recall, and onset F-measure (± 50 ms).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.status import Result

CONDITIONS = ("mixture_phone", "mixture_karaoke", "separated", "clean", "trimmed", "multi_singer")


@dataclass
class RealItem:
    id: str
    audio: Path
    role: str  # "user" | "target"
    condition: str
    singer: str = ""
    session: str = ""
    target: str | None = None
    lyrics: str | None = None
    annotations: dict = field(default_factory=dict)
    recording: dict = field(default_factory=dict)


@dataclass
class RealSet:
    root: Path
    items: dict[str, RealItem]

    @property
    def users(self) -> list[RealItem]:
        return [i for i in self.items.values() if i.role == "user"]


def load_realset(folder: str | Path) -> Result[RealSet]:
    root = Path(folder)
    man = root / "manifest.jsonl"
    if not man.is_file():
        return Result.failure(f"no manifest.jsonl in {root}")
    items: dict[str, RealItem] = {}
    errors: list[str] = []
    for n, line in enumerate(man.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {n}: {exc}")
            continue
        missing = [k for k in ("id", "audio", "role", "condition") if k not in d]
        if missing:
            errors.append(f"line {n}: missing {missing}")
            continue
        if d["condition"] not in CONDITIONS:
            errors.append(f"line {n}: unknown condition {d['condition']!r} (one of {CONDITIONS})")
            continue
        if d["role"] not in ("user", "target"):
            errors.append(f"line {n}: role must be 'user' or 'target'")
            continue
        path = root / d["audio"]
        if not path.is_file():
            errors.append(f"line {n}: audio file not found: {d['audio']}")
            continue
        if d["id"] in items:
            errors.append(f"line {n}: duplicate id {d['id']!r}")
            continue
        items[d["id"]] = RealItem(d["id"], path, d["role"], d["condition"], d.get("singer", ""), d.get("session", ""), d.get("target"),
                                  d.get("lyrics"), d.get("annotations") or {}, d.get("recording") or {})
    for it in items.values():
        if it.role == "user":
            if not it.singer:
                errors.append(f"{it.id}: user recordings need a singer (the split is by singer)")
            if not it.target or it.target not in items or items[it.target].role != "target":
                errors.append(f"{it.id}: target {it.target!r} is not a target recording in the manifest")
    if errors:
        return Result.failure("manifest problems: " + "; ".join(errors[:20]) + (" …" if len(errors) > 20 else ""))
    return Result.success(RealSet(root, items))


def split_realset(rs: RealSet, held_out_fraction: float = 0.3, seed: int = 0, path: str | Path | None = None) -> dict[str, str]:
    """Singer-disjoint (hence session-disjoint) split → {user id: "tuning" | "held_out"}; reused from ``split.json`` if present."""
    path = Path(path) if path else rs.root / "split.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if set(saved) >= {u.id for u in rs.users}:
            return {u.id: saved[u.id] for u in rs.users}
    singers = sorted({u.singer for u in rs.users})
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(singers))
    n_hold = max(1, int(round(held_out_fraction * len(singers)))) if len(singers) > 1 else 0
    held = set(order[:n_hold])
    sessions = defaultdict(set)
    for u in rs.users:
        sessions[u.session or u.id].add(u.singer)
    shared = [s for s, who in sessions.items() if len(who & held) and len(who - held)]
    if shared:
        raise ValueError(f"sessions recorded by singers on both sides of the split: {shared}")
    out = {u.id: ("held_out" if u.singer in held else "tuning") for u in rs.users}
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------- annotations


def _read_f0(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t, f = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            try:
                t.append(float(row[0]))
                f.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return np.array(t), np.array(f)


def _onsets(ann, root: Path) -> np.ndarray | None:
    if ann is None:
        return None
    if isinstance(ann, (list, tuple)):
        return np.asarray(ann, float)
    p = root / ann
    return np.array([float(v) for v in p.read_text(encoding="utf-8").split()]) if p.is_file() else None


def pitch_agreement(rep, times: np.ndarray, f0_hz: np.ndarray) -> dict[str, float]:
    """RPA / RCA / voicing recall of ``rep``'s f0 against an annotated track."""
    est_c = rep.curves["f0_cents"].values
    gt = rep.grid.times()
    voiced = f0_hz > 0
    if not voiced.any():
        return {}
    e = np.interp(times[voiced], gt, np.nan_to_num(est_c, nan=np.nan), left=np.nan, right=np.nan)
    ref = 1200 * np.log2(f0_hz[voiced] / 440.0)
    found = np.isfinite(e)
    d = e - ref
    chroma = (d + 600) % 1200 - 600
    return {"rpa": float(np.mean(found & (np.abs(np.nan_to_num(d, nan=1e9)) <= 50))),
            "rca": float(np.mean(found & (np.abs(np.nan_to_num(chroma, nan=1e9)) <= 50))),
            "voicing_recall": float(np.mean(found))}


def onset_f_measure(est: np.ndarray, ref: np.ndarray, tol_s: float = 0.05) -> float:
    est, ref = np.sort(np.asarray(est, float)), np.sort(np.asarray(ref, float))
    if not len(est) and not len(ref):
        return 1.0
    used = np.zeros(len(est), bool)
    hits = 0
    for r in ref:
        cand = np.flatnonzero(~used & (np.abs(est - r) <= tol_s))
        if cand.size:
            used[cand[np.argmin(np.abs(est[cand] - r))]] = True
            hits += 1
    p = hits / max(len(est), 1)
    rcl = hits / max(len(ref), 1)
    return float(2 * p * rcl / (p + rcl)) if p + rcl else 0.0


# ---------------------------------------------------------------- evaluation


@dataclass
class ConditionReport:
    condition: str
    n: int = 0
    failed: int = 0
    octave_decided: int = 0
    octave_withheld: int = 0
    octave_errors: int = 0
    octave_annotated: int = 0
    rhythm_verdicts: int = 0
    false_rhythm_verdicts: int = 0
    judgements: int = 0
    withheld: int = 0
    premise_failures: dict = field(default_factory=dict)
    rpa: list = field(default_factory=list)
    rca: list = field(default_factory=list)
    voicing_recall: list = field(default_factory=list)
    onset_f: list = field(default_factory=list)
    separated: int = 0

    def summary(self) -> dict:
        m = lambda v: float(np.mean(v)) if v else None  # noqa: E731
        return {"condition": self.condition, "n": self.n, "failed": self.failed, "separated": self.separated,
                "octave_decided": self.octave_decided, "octave_withheld": self.octave_withheld, "octave_errors": self.octave_errors,
                "octave_error_rate": self.octave_errors / self.octave_decided if self.octave_decided else None,
                "rhythm_verdicts": self.rhythm_verdicts, "false_rhythm_verdicts": self.false_rhythm_verdicts,
                "false_rhythm_rate": self.false_rhythm_verdicts / self.rhythm_verdicts if self.rhythm_verdicts else None,
                "withheld_rate": self.withheld / (self.withheld + self.judgements) if self.withheld + self.judgements else None,
                "premise_failures": self.premise_failures, "rpa": m(self.rpa), "rca": m(self.rca),
                "voicing_recall": m(self.voicing_recall), "onset_f": m(self.onset_f)}


@dataclass
class RealsetReport:
    split: str
    conditions: dict[str, ConditionReport]
    failures: list[str]
    settings: dict

    def table(self) -> str:
        cols = ("n", "octave_decided", "octave_withheld", "octave_errors", "false_rhythm_verdicts", "withheld_rate", "rpa", "rca", "onset_f")
        lines = [f"{'condition':18s} " + " ".join(f"{c[:14]:>14s}" for c in cols)]
        for name, cr in sorted(self.conditions.items()):
            s = cr.summary()
            lines.append(f"{name:18s} " + " ".join(f"{'-' if s[c] is None else (f'{s[c]:.3f}' if isinstance(s[c], float) else str(s[c])):>14s}" for c in cols))
        return "\n".join(lines)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        data = {"split": self.split, "settings": self.settings, "failures": self.failures,
                "conditions": {k: v.summary() for k, v in self.conditions.items()}}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        return path


def evaluate_realset(rs: RealSet, *, split: str = "all", held_out_fraction: float = 0.3, seed: int = 0, separation: str = "auto",
                     trackers=None, rhythm_verdict_ms: float = 50.0, on_time_tolerance_ms: float = 30.0,
                     analyze_fn: Callable | None = None, explain_config=None, progress: Callable[[str], None] | None = None) -> RealsetReport:
    from ..attributes.extract import analyze
    from ..core.consent import Provenance
    from ..core.containers import Recording
    from ..explain import explain
    from ..io import load_audio

    if split not in ("all", "tuning", "held_out"):
        raise ValueError("split must be 'all', 'tuning' or 'held_out'")
    sides = split_realset(rs, held_out_fraction, seed) if split != "all" else {u.id: "all" for u in rs.users}
    users = [u for u in rs.users if split == "all" or sides[u.id] == split]

    def run(item: RealItem, provenance):
        x, sr = load_audio(item.audio)
        backing = None
        if item.recording.get("backing"):
            b, bsr = load_audio(rs.root / item.recording["backing"])
            from ..dsp.base import resample

            backing = resample(b, bsr, sr)
        rec = Recording(x, sr, provenance, owner_id=item.singer or "realset" if provenance is Provenance.USER else None)
        if analyze_fn is not None:
            return analyze_fn(rec, backing=backing, separation=separation)
        return analyze(rec, trackers=trackers, backing=backing, separation=separation)

    target_cache: dict[str, object] = {}
    conds: dict[str, ConditionReport] = {}
    failures: list[str] = []
    for u in users:
        cr = conds.setdefault(u.condition, ConditionReport(u.condition))
        cr.n += 1
        if progress:
            progress(u.id)
        if u.target not in target_cache:
            target_cache[u.target] = run(rs.items[u.target], Provenance.REFERENCE)
        tr = target_cache[u.target]
        ur = run(u, Provenance.USER)
        if not (tr.usable and ur.usable):
            cr.failed += 1
            failures.append(f"{u.id}: analysis failed ({tr.reason or ur.reason})")
            continue
        rep, ann = ur.value, u.annotations
        if (rep.quality or {}).get("separation", {}).get("applied"):
            cr.separated += 1
        if ann.get("f0"):
            t, f = _read_f0(rs.root / ann["f0"])
            for k, v in pitch_agreement(rep, t, f).items():
                getattr(cr, k).append(v)
        ons = _onsets(ann.get("onsets"), rs.root)
        if ons is not None:
            est = np.array([s for s, _ in rep.meta.get("notes", [])]) * rep.grid.hop_seconds
            cr.onset_f.append(onset_f_measure(est, ons))
        ex = explain([rep], tr.value, explain_config, lyrics=u.lyrics) if explain_config is not None else explain([rep], tr.value, lyrics=u.lyrics)
        if not ex.usable:
            cr.failed += 1
            failures.append(f"{u.id}: explain failed ({ex.reason})")
            continue
        e = ex.value
        oct_p = e.premises.get("octave_relation")
        ann_oct = ann.get("octave_relation")
        if ann_oct is None and "transposition_semitones" in ann:
            ann_oct = round(ann["transposition_semitones"] / 12)
        if oct_p is not None and oct_p.holds and e.transposition_cents is not None:
            cr.octave_decided += 1
            if ann_oct is not None:
                cr.octave_annotated += 1
                cr.octave_errors += int(round(e.transposition_cents / 1200) != int(ann_oct))
        else:
            cr.octave_withheld += 1
        # true onset deviation per target note (ms); a verdict is false when the truth is within tolerance
        all_on_time = ann.get("rhythm_ok") is True
        true_dev = {int(k): float(v) for k, v in ann.get("onset_deviation_ms", {}).items()} if isinstance(ann.get("onset_deviation_ms"), dict) else {}
        for it in e.items:
            if it.category != "rhythm":
                continue
            verdict = (abs(it.magnitude) >= rhythm_verdict_ms) if it.unit == "ms" else (abs(it.magnitude) >= 5.0)
            if not verdict:
                continue
            cr.rhythm_verdicts += 1
            k = int(it.detail.get("target_note", -1))
            if all_on_time:
                cr.false_rhythm_verdicts += 1
            elif it.unit == "ms" and k in true_dev:
                # content-aligned timing is relative to the previous note: compare with the relative truth
                truth = true_dev[k] - true_dev.get(k - 1, 0.0) if it.detail.get("reference") == "previous note" else true_dev[k]
                cr.false_rhythm_verdicts += int(abs(truth) < on_time_tolerance_ms)
        cr.judgements += len(e.items)
        cr.withheld += len(e.withheld)
        for name, p in e.premises.items():
            if not p.holds:
                cr.premise_failures[name] = cr.premise_failures.get(name, 0) + 1
    settings = {"separation": separation, "rhythm_verdict_ms": rhythm_verdict_ms, "on_time_tolerance_ms": on_time_tolerance_ms,
                "held_out_fraction": held_out_fraction, "seed": seed}
    return RealsetReport(split, conds, failures, settings)


def tracker_breakdown(rs: RealSet, trackers, *, split: str = "all", held_out_fraction: float = 0.3, seed: int = 0) -> dict[str, dict[str, dict[str, float]]]:
    """Per-condition, per-tracker pitch agreement on the *raw* input (revision A2 tuning aid).

    Shows how each tracker of the consensus holds up on mixtures versus
    separated or clean vocals, independent of separation and consensus.
    → ``{condition: {tracker: {"rpa", "rca", "voicing_recall", "n"}}}`` (means over items).
    """
    from ..io import load_audio

    sides = split_realset(rs, held_out_fraction, seed) if split != "all" else {u.id: "all" for u in rs.users}
    acc: dict[str, dict[str, dict[str, list[float]]]] = {}
    for u in rs.users:
        if split != "all" and sides[u.id] != split or not u.annotations.get("f0"):
            continue
        t, f = _read_f0(rs.root / u.annotations["f0"])
        voiced = f > 0
        if not voiced.any():
            continue
        x, sr = load_audio(u.audio)
        ref = 1200 * np.log2(f[voiced] / 440.0)
        for tr in trackers:
            r = tr.track(x, sr)
            if not r.usable:
                continue
            pt = r.value
            c = 1200 * np.log2(np.where(pt.f0_hz > 0, pt.f0_hz, np.nan) / 440.0)
            e = np.interp(t[voiced], pt.times, c, left=np.nan, right=np.nan)
            found = np.isfinite(e)
            d = np.nan_to_num(e - ref, nan=1e9)
            chroma = np.where(found, (d + 600) % 1200 - 600, 1e9)
            m = acc.setdefault(u.condition, {}).setdefault(tr.name, {"rpa": [], "rca": [], "voicing_recall": []})
            m["rpa"].append(float(np.mean(found & (np.abs(d) <= 50))))
            m["rca"].append(float(np.mean(found & (np.abs(chroma) <= 50))))
            m["voicing_recall"].append(float(np.mean(found)))
    return {c: {name: {**{k: float(np.mean(v)) for k, v in m.items()}, "n": len(m["rpa"])} for name, m in by.items()}
            for c, by in acc.items()}


__all__ = ["CONDITIONS", "ConditionReport", "RealItem", "RealSet", "RealsetReport", "evaluate_realset", "load_realset",
           "onset_f_measure", "pitch_agreement", "split_realset", "tracker_breakdown"]

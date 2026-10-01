"""Consent-aware storage for voice-derived data (reference service layer, revision C1).

User state — consent records, stored features, raw audio and their deletion —
lives here, outside the stateless ``gyeol`` library.  The library keeps the
consent *types and guards* (:class:`gyeol.core.ConsentToken`,
:class:`gyeol.core.ConsentedVoice`, :func:`gyeol.core.require_consented_voice`).

Singer vectors, embeddings and attribute features derived from a voice are
treated as sensitive biometric information (PIPA).  This module provides

* :class:`ConsentStore` – per-user, per-purpose consent flags
  (``analysis``, ``training``, ``storage``, ``voice_synthesis``) with history;
* :class:`FeatureStore` – stores feature arrays only with ``storage`` consent;
* :class:`RawAudioStore` + :class:`RetentionPolicy` – raw audio is deleted
  right after feature extraction by default;
* :func:`delete_user` – removes every stored artefact of a user and revokes
  their consent, leaving only a non-biometric deletion log entry.

The filesystem layout is ``<root>/consent/<user>.json`` and
``<root>/users/<user>/{features,raw}/``.  User ids are validated so they
cannot escape the root directory.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from gyeol.core.consent import ConsentError, ConsentToken, Purpose

_USER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _check_id(value: str, what: str = "user id") -> str:
    if not _USER_RE.match(value) or value in (".", ".."):
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class ConsentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, user_id: str) -> Path:
        return self.root / "consent" / f"{_check_id(user_id)}.json"

    def _load(self, user_id: str) -> dict:
        p = self._path(user_id)
        return json.loads(p.read_text()) if p.exists() else {"user_id": user_id, "purposes": {}, "tokens": {}, "history": []}

    def _save(self, user_id: str, rec: dict) -> None:
        _atomic_write(self._path(user_id), json.dumps(rec, ensure_ascii=False, indent=1).encode())

    def grant(self, user_id: str, purposes: set[Purpose] | list[Purpose]) -> ConsentToken:
        rec = self._load(user_id)
        ps = frozenset(Purpose(p) for p in purposes)
        now = time.time()
        for p in ps:
            rec["purposes"][p.value] = True
        tok = ConsentToken(user_id=user_id, purposes=ps, issued_at=now)
        rec["tokens"][tok.token_id] = {"purposes": sorted(p.value for p in ps), "issued_at": now, "revoked": False}
        rec["history"].append({"t": now, "action": "grant", "purposes": sorted(p.value for p in ps)})
        self._save(user_id, rec)
        return tok

    def revoke(self, user_id: str, purpose: Purpose) -> None:
        rec = self._load(user_id)
        purpose = Purpose(purpose)
        rec["purposes"][purpose.value] = False
        for t in rec["tokens"].values():
            if purpose.value in t["purposes"]:
                t["revoked"] = True
        rec["history"].append({"t": time.time(), "action": "revoke", "purposes": [purpose.value]})
        self._save(user_id, rec)

    def allows(self, user_id: str, purpose: Purpose) -> bool:
        return bool(self._load(user_id)["purposes"].get(Purpose(purpose).value, False))

    def require(self, user_id: str, purpose: Purpose) -> None:
        if not self.allows(user_id, purpose):
            raise ConsentError(f"user {user_id!r} has not consented to {Purpose(purpose).value}")

    def verify(self, token: ConsentToken) -> bool:
        """True if the token was issued by this store, is not revoked and still granted."""
        rec = self._load(token.user_id)
        t = rec["tokens"].get(token.token_id)
        if t is None or t["revoked"] or token.revoked:
            return False
        return all(rec["purposes"].get(p.value, False) for p in token.purposes)

    def forget(self, user_id: str) -> None:
        p = self._path(user_id)
        if p.exists():
            p.unlink()


class RawAudioRetention(str, Enum):
    DELETE_AFTER_EXTRACTION = "delete_after_extraction"
    KEEP_DAYS = "keep_days"
    KEEP = "keep"


@dataclass(frozen=True)
class RetentionPolicy:
    raw_audio: RawAudioRetention = RawAudioRetention.DELETE_AFTER_EXTRACTION
    keep_days: float = 0.0

    def expired(self, stored_at: float, now: float) -> bool:
        if self.raw_audio is RawAudioRetention.KEEP:
            return False
        if self.raw_audio is RawAudioRetention.DELETE_AFTER_EXTRACTION:
            return True
        return now - stored_at > self.keep_days * 86400.0


class FeatureStore:
    """Voice-derived arrays (singer vectors, embeddings, curves) per user."""

    def __init__(self, root: str | Path, consent: ConsentStore):
        self.root = Path(root)
        self.consent = consent

    def _dir(self, user_id: str) -> Path:
        return self.root / "users" / _check_id(user_id) / "features"

    def put(self, user_id: str, key: str, arrays: dict[str, np.ndarray], meta: dict | None = None) -> Path:
        self.consent.require(user_id, Purpose.STORAGE)
        path = self._dir(user_id) / f"{_check_id(key, 'key')}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(arrays)
        payload["__meta__"] = np.frombuffer(json.dumps(meta or {}, ensure_ascii=False).encode(), dtype=np.uint8)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
        return path

    def get(self, user_id: str, key: str) -> tuple[dict[str, np.ndarray], dict]:
        path = self._dir(user_id) / f"{_check_id(key, 'key')}.npz"
        with np.load(path) as z:
            meta = json.loads(bytes(z["__meta__"]).decode())
            return {k: z[k] for k in z.files if k != "__meta__"}, meta

    def keys(self, user_id: str) -> list[str]:
        d = self._dir(user_id)
        return sorted(p.stem for p in d.glob("*.npz")) if d.exists() else []


class RawAudioStore:
    def __init__(self, root: str | Path, consent: ConsentStore, policy: RetentionPolicy | None = None):
        self.root = Path(root)
        self.consent = consent
        self.policy = policy or RetentionPolicy()

    def _dir(self, user_id: str) -> Path:
        return self.root / "users" / _check_id(user_id) / "raw"

    def put(self, user_id: str, recording_id: str, audio: np.ndarray, sr: int) -> Path | None:
        """Store raw audio if the policy keeps it and the user consented to storage.

        Returns None (nothing written) under ``DELETE_AFTER_EXTRACTION``.
        """
        if self.policy.raw_audio is RawAudioRetention.DELETE_AFTER_EXTRACTION:
            return None
        self.consent.require(user_id, Purpose.STORAGE)
        path = self._dir(user_id) / f"{_check_id(recording_id, 'recording id')}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, audio=np.asarray(audio, np.float32), sr=np.int64(sr), stored_at=np.float64(time.time()))
        return path

    def after_extraction(self, user_id: str, recording_id: str) -> bool:
        """Call once features are extracted; deletes raw audio when the policy says so."""
        path = self._dir(user_id) / f"{_check_id(recording_id, 'recording id')}.npz"
        if path.exists() and self.policy.raw_audio is RawAudioRetention.DELETE_AFTER_EXTRACTION:
            path.unlink()
            return True
        return False

    def sweep(self, now: float | None = None) -> int:
        """Delete expired raw audio for all users.  Returns the number removed."""
        now = time.time() if now is None else now
        n = 0
        for path in (self.root / "users").glob("*/raw/*.npz"):
            with np.load(path) as z:
                stored_at = float(z["stored_at"])
            if self.policy.expired(stored_at, now):
                path.unlink()
                n += 1
        return n


def delete_user(root: str | Path, user_id: str) -> dict[str, int]:
    """Delete all stored data of ``user_id`` (features, raw audio, consent record).

    A deletion-log line (user id and timestamp only, no voice data) is kept so
    the deletion itself can be demonstrated.
    """
    root = Path(root)
    user_dir = root / "users" / _check_id(user_id)
    counts = {"files": 0}
    if user_dir.exists():
        counts["files"] = sum(1 for p in user_dir.rglob("*") if p.is_file())
        shutil.rmtree(user_dir)
    ConsentStore(root).forget(user_id)
    log = root / "deletion_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"user_id": user_id, "deleted_at": time.time(), "files": counts["files"]}) + "\n")
    return counts

"""Bakes fantômes (21/09) : étiqueter des prompts pendant le temps mort GPU.

Spec : docs/superpowers/specs/2026-09-21-bakes-fantomes-design.md.

Sous V1 un cycle de fenêtre dure ~687 s (médiane) et le mineur ne produit que
~263 s : le GPU dort pendant le trou 503. Le hors-zone dépend surtout de « ce
prompt a-t-il été mesuré récemment, et avec quel verdict » (dernière mesure en
zone : ~21 % de jetés ; jamais mesuré : ~39 % ; déjà hors zone : ~60 %). On
utilise ce temps mort pour fabriquer ces mesures sur la bande p95-p99 du prior.

Ce module ne contient que des fonctions PURES (drapeaux, fenêtre de tir,
randomness synthétique, choix des cibles, étiquette). Le branchement dans la
boucle de génération vit dans ``engine.MiningEngine._ghost_ready`` /
``_ghost_lot``.
"""
from __future__ import annotations

import hashlib
import json as _json
import os as _os
import random as _random
from typing import Iterable, Sequence

GHOST_T_MIN_DEFAULT = 330.0
GHOST_T_MAX_DEFAULT = 537.0
GHOST_LOT_DEFAULT = 8
GHOST_REFRESH_AFTER = 250


def ghost_enabled() -> bool:
    """``RELIQUARY_GHOST_BAKE=1`` arme la boucle. Défaut : éteinte."""
    return _os.environ.get("RELIQUARY_GHOST_BAKE", "0") == "1"


def ghost_feed_enabled() -> bool:
    """``RELIQUARY_GHOST_FEED=1`` : les étiquettes alimentent le mémo et la
    liste noire. Défaut 0 = étape A (on mesure seulement la non-perturbation)."""
    return _os.environ.get("RELIQUARY_GHOST_FEED", "0") == "1"


def _env_float(name: str, default: float) -> float:
    try:
        return float(_os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def ghost_bounds() -> tuple[float, float]:
    """Fenêtre de tir, en secondes depuis l'ouverture de la fenêtre courante.

    330 s ≈ fin de notre production ; 537 s = 627 s (cycle le plus court
    observé sur 349 fenêtres) − 90 s de marge avant le flip le plus précoce."""
    return (_env_float("RELIQUARY_GHOST_T_MIN", GHOST_T_MIN_DEFAULT),
            _env_float("RELIQUARY_GHOST_T_MAX", GHOST_T_MAX_DEFAULT))


def ghost_lot_size() -> int:
    try:
        return max(1, int(_os.environ.get("RELIQUARY_GHOST_LOT", "") or GHOST_LOT_DEFAULT))
    except (TypeError, ValueError):
        return GHOST_LOT_DEFAULT


def ghost_window_ok(elapsed: float | None, *, t_min: float, t_max: float) -> bool:
    """Vrai si l'on est dans la fenêtre de tir. Ouverture inconnue => jamais."""
    return elapsed is not None and t_min <= elapsed <= t_max


def synthetic_randomness(window_n, lot_seq: int) -> str:
    """Randomness de génération fantôme : hash préfixé par un domaine propre,
    impossible à confondre avec celle d'une vraie fenêtre (un groupe fantôme ne
    peut donc jamais passer pour soumettable, même par erreur)."""
    return hashlib.sha256(
        f"reliquary-ghost|{window_n}|{lot_seq}".encode()).hexdigest()


def band_indices(scores, *, lo_pct: float = 95.0, hi_pct: float = 99.0):
    """Indices des prompts dont le score est dans la bande [p_lo, p_hi]."""
    import numpy as np
    s = np.asarray(scores, dtype=float)
    lo, hi = np.percentile(s, [lo_pct, hi_pct])
    return np.nonzero((s >= lo) & (s <= hi))[0]


class GhostTargets:
    """Réservoir des cibles fantômes (la bande), tirées au hasard à chaque lot."""

    def __init__(self, band: Iterable[int], *, rng: _random.Random | None = None):
        self._band = [int(i) for i in band]
        self._rng = rng or _random.Random()

    def __len__(self) -> int:
        return len(self._band)

    def pick(self, n: int, *, exclude: set, seen: dict, now_window: int,
             refresh_after: int = GHOST_REFRESH_AFTER) -> list[int]:
        """Jusqu'à ``n`` cibles hors ``exclude`` et non étiquetées depuis moins
        de ``refresh_after`` fenêtres."""
        now = int(now_window or 0)
        cand = [i for i in self._band
                if i not in exclude
                and (i not in seen or now - int(seen[i]) > refresh_after)]
        if len(cand) <= n:
            return cand
        return self._rng.sample(cand, n)


def label_in_zone(rewards: Sequence[float], timeouts: Sequence[bool]) -> bool:
    """Étiquette « en zone » calculée EXACTEMENT comme le filtre de production
    (``_skip_for_out_of_zone`` sur les timeouts imputés)."""
    from reliquary.miner import engine as _e   # paresseux : engine importe ce module
    return not _e._skip_for_out_of_zone(
        _e.timeout_imputed_for_zone(list(rewards), list(timeouts)))


def is_truncated(completion: Sequence[int], *, eos_ids: Iterable[int]) -> bool:
    """Un rollout sans EOS est tronqué (arrêté par la limite de tokens)."""
    eos = set(int(x) for x in eos_ids)
    return not any(int(t) in eos for t in completion)


def ghost_seen_from_dump(path: str) -> dict[int, int]:
    """Mémoire « déjà étiqueté » reconstruite depuis ``RELIQUARY_GHOST_DUMP``
    (prompt -> dernière fenêtre d'étiquetage). Sans elle, chaque restart
    ré-étiquetait les mêmes prompts. Jamais d'exception."""
    seen: dict[int, int] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    r = _json.loads(line)
                    idx, w = int(r["prompt_idx"]), r.get("window_n")
                    if w is None:
                        continue
                    seen[idx] = max(seen.get(idx, -1), int(w))
                except Exception:
                    continue
    except Exception:
        return {}
    return seen

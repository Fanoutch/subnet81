"""Mémo des payables mesurés (2026-08-18) — le « slot mémo » du sprint.

Les tranches (5 000 sur ~2,4 M, offset sha256(randomness) glissant) recyclent
les prompts : un prompt donné retombe ~1 fenêtre sur 480, et notre stock de
payables mesurés (~4 000, +1 000/jour) en place ~4-8 par tranche. Mesures :
1 083 réapparitions d'ex-payables en 3 jours, 51 % encore payables, 34 %
jamais re-générées ; banc armement 69 % → 75 % en donnant le 3e slot du
sprint au meilleur ex-payable de la tranche.

La table vit en mémoire : chargée au boot depuis le dump JSONL du mineur
(``RELIQUARY_SAMPLE_DUMP``), maintenue par ``dump_group_sample`` (le puits
central du grading). LA DERNIÈRE MESURE FAIT FOI (churn 41 %/qq heures : un
ex-payable re-mesuré k=8 sort de la table). Le cooldown validateur n'a pas à
être géré ici : le classement de tranche l'exclut en amont.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class PayableMemo:
    def __init__(self) -> None:
        self._seq = 0
        self._payable: dict[int, int] = {}   # prompt_idx -> seq de la mesure
        # Mémo de tête (04/09) : confirmations consécutives en zone et
        # dernière fenêtre de mesure, pour classer les candidats.
        self._conf: dict[int, int] = {}
        self._last_w: dict[int, int] = {}

    def size(self) -> int:
        return len(self._payable)

    def clear(self) -> None:
        self._payable.clear()
        self._conf.clear()
        self._last_w.clear()
        self._seq = 0

    def update(self, prompt_idx: int, payable: bool,
               window_n: int | None = None) -> None:
        self._seq += 1
        idx = int(prompt_idx)
        if payable:
            self._payable[idx] = self._seq
            self._conf[idx] = self._conf.get(idx, 0) + 1
            if window_n is not None:
                self._last_w[idx] = max(self._last_w.get(idx, 0), int(window_n))
        else:
            self._payable.pop(idx, None)
            self._conf.pop(idx, None)
            self._last_w.pop(idx, None)

    def load_jsonl(self, path: str) -> None:
        """Charge l'historique. Jamais d'exception (fichier absent/corrompu)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                n = 0
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    idx = r.get("prompt_idx")
                    if idx is None:
                        continue
                    payable = bool(r.get("in_zone")) and \
                        int(r.get("n_truncated", 0) or 0) == 0
                    self.update(int(idx), payable, window_n=r.get("window_n"))
                    n += 1
            logger.info(
                "payable_memo: %d lignes chargées, %d payables connus (%s)",
                n, len(self._payable), path,
            )
        except FileNotFoundError:
            logger.info("payable_memo: pas d'historique (%s)", path)
        except Exception:
            logger.exception("payable_memo: chargement échoué (non fatal)")

    def best_in_range(self, lo: int, hi: int,
                      exclude: set[int] | None = None) -> int | None:
        """L'ex-payable LE PLUS FRAIS de [lo, hi), hors ``exclude``."""
        exclude = exclude or set()
        best, best_seq = None, -1
        for idx, seq in self._payable.items():
            if lo <= idx < hi and idx not in exclude and seq > best_seq:
                best, best_seq = idx, seq
        return best


    def top_in_range(self, lo: int, hi: int,
                     exclude: set[int] | None = None, n: int = 1,
                     run_start: int = 0) -> list[int]:
        """Les ``n`` meilleurs ex-payables de [lo, hi), hors ``exclude``.

        Ordre : mesuré dans le run courant (dernière fenêtre ≥ ``run_start``)
        d'abord, puis nombre de confirmations, puis fraîcheur. Mesuré 04/09 :
        zone→zone 90 % (run courant) / 86 % (run précédent) / 77 % (ère v4)
        contre 67 % pour un pick classé jamais vu.
        """
        exclude = exclude or set()
        cands = [
            idx for idx in self._payable
            if lo <= idx < hi and idx not in exclude
        ]
        cands.sort(key=lambda i: (
            1 if (run_start and self._last_w.get(i, 0) >= run_start) else 0,
            self._conf.get(i, 0),
            self._payable[i],
        ), reverse=True)
        return cands[:max(0, int(n))]


_MEMO = PayableMemo()


def get_memo() -> PayableMemo:
    return _MEMO

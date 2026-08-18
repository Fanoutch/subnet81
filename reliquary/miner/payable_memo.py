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

    def size(self) -> int:
        return len(self._payable)

    def clear(self) -> None:
        self._payable.clear()
        self._seq = 0

    def update(self, prompt_idx: int, payable: bool) -> None:
        self._seq += 1
        if payable:
            self._payable[int(prompt_idx)] = self._seq
        else:
            self._payable.pop(int(prompt_idx), None)

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
                    self.update(int(idx), payable)
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


_MEMO = PayableMemo()


def get_memo() -> PayableMemo:
    return _MEMO

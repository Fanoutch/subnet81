"""Carnet de curseurs du mineur SFT (corpus), partagé entre deux fils.

Le validateur corpus n'accepte une soumission que pour le curseur qu'il attend
de la hotkey (``bad_cursor`` sinon) : les soumissions sont STRICTEMENT
séquentielles. Mais le prompt de chaque curseur est fixé d'avance
(``reliquary.corpus.walk.walk_index``) : on peut donc générer les curseurs
c, c+1, ..., c+K en parallèle et ne soumettre que dans l'ordre.

- fil GPU : ``next_admission`` → génère → ``on_generated`` / ``on_failed`` ;
  ``pop_aborts`` rend les curseurs en vol devenus inutiles.
- fil de soumission : ``take_submittable`` → POST → ``on_accepted`` ou, après
  tout refus, ``realign(curseur relu sur le validateur)``.

Aucune dépendance au protocole ici : logique pure, testée sans GPU.
"""
from __future__ import annotations

import threading


class CursorBook:
    def __init__(self, *, start: int, max_in_flight: int, max_ahead: int) -> None:
        if max_in_flight < 1 or max_ahead < 1:
            raise ValueError("max_in_flight et max_ahead doivent valoir au moins 1")
        self._lock = threading.Lock()
        self.max_in_flight = max_in_flight
        self.max_ahead = max_ahead
        self._next_submit = start        # le curseur que le validateur attend
        self._admit = start              # prochain curseur jamais encore admis
        self._in_flight: set[int] = set()
        self._ready: dict[int, object] = {}
        self._readmit: list[int] = []    # à régénérer, prioritaires
        self._submitting: int | None = None
        self._aborts: list[int] = []
        self.stopped: str | None = None

    # --- fil GPU -----------------------------------------------------------
    def next_admission(self) -> int | None:
        """Le prochain curseur à générer, ou None si la fenêtre est pleine."""
        with self._lock:
            if self.stopped or len(self._in_flight) >= self.max_in_flight:
                return None
            if self._readmit:
                cursor = self._readmit.pop(0)
            elif self._admit - self._next_submit < self.max_ahead:
                cursor = self._admit
                self._admit += 1
            else:
                return None
            self._in_flight.add(cursor)
            return cursor

    def on_generated(self, cursor: int, generations) -> None:
        with self._lock:
            if cursor not in self._in_flight:
                return                   # annulé entre-temps (réalignement)
            self._in_flight.discard(cursor)
            if cursor >= self._next_submit:
                self._ready[cursor] = generations

    def on_failed(self, cursor: int) -> None:
        """Génération inutilisable (préemption KV : preuve impossible)."""
        with self._lock:
            if cursor not in self._in_flight:
                return
            self._in_flight.discard(cursor)
            if cursor >= self._next_submit and cursor not in self._readmit:
                self._readmit.append(cursor)
                self._readmit.sort()

    def pop_aborts(self) -> list[int]:
        with self._lock:
            out, self._aborts = self._aborts, []
            return out

    # --- fil de soumission -------------------------------------------------
    def take_submittable(self):
        """(curseur, générations) du curseur attendu s'il est prêt, sinon None."""
        with self._lock:
            if self.stopped or self._submitting is not None:
                return None
            cursor = self._next_submit
            if cursor not in self._ready:
                return None
            self._submitting = cursor
            return cursor, self._ready.pop(cursor)

    def on_accepted(self, cursor: int) -> None:
        with self._lock:
            self._submitting = None
            if cursor == self._next_submit:
                self._next_submit += 1

    def realign(self, cursor: int) -> None:
        """Se caler sur le curseur que le validateur attend réellement."""
        with self._lock:
            self._submitting = None
            self._next_submit = cursor
            for c in [c for c in self._ready if c < cursor]:
                del self._ready[c]
            stale = sorted(c for c in self._in_flight if c < cursor)
            for c in stale:
                self._in_flight.discard(c)
            self._aborts.extend(stale)
            self._readmit = [c for c in self._readmit if c >= cursor]
            if self._admit < cursor:
                self._admit = cursor
            for c in range(cursor, self._admit):
                if c not in self._ready and c not in self._in_flight and c not in self._readmit:
                    self._readmit.append(c)
            self._readmit.sort()

    def stop(self, reason: str) -> None:
        with self._lock:
            self.stopped = reason

    # --- lecture -----------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {"next_submit": self._next_submit, "in_flight": len(self._in_flight),
                    "ready": len(self._ready), "readmit": len(self._readmit),
                    "admit": self._admit}

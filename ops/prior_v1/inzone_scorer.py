"""Scoreur « en zone » V1 sans scikit-learn (réplique de TfidfVectorizer + LogReg).

Modèle exporté par train_inzone_v1.py : vocabulaire (n-grammes 1-2), idf,
coefficients, intercept, poids de la longueur. Reproduit EXACTEMENT
TfidfVectorizer(lowercase, token_pattern (?u)\\b\\w\\w+\\b, ngram (1,2),
sublinear_tf, norm l2) suivi du produit scalaire logistique.
"""
from __future__ import annotations

import json
import math
import re

PRE = "Solve the following programming problem step by step.\n\n"
_TOK = re.compile(r"(?u)\b\w\w+\b")


class InZoneScorer:
    def __init__(self, path: str) -> None:
        m = json.load(open(path))
        self.vocab = m["vocab"]              # terme -> colonne
        self.idf = m["idf"]                  # liste par colonne
        self.coef = m["coef"]                # liste par colonne
        self.len_coef = m["len_coef"]
        self.intercept = m["intercept"]

    def proba(self, text: str) -> float:
        t = text[len(PRE):] if text.startswith(PRE) else text
        words = _TOK.findall(t.lower())
        counts: dict = {}
        for i, w in enumerate(words):
            j = self.vocab.get(w)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
            if i + 1 < len(words):
                j = self.vocab.get(w + " " + words[i + 1])
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
        vals = {j: (1.0 + math.log(c)) * self.idf[j] for j, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vals.values())) or 1.0
        z = self.intercept + self.len_coef * (math.log1p(len(t)) / 10.0)
        for j, v in vals.items():
            z += self.coef[j] * v / norm
        return 1.0 / (1.0 + math.exp(-z))

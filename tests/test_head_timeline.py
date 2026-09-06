"""Instrumentation de la 2e tête (06/09) : livraison → prise en charge.

Le dump n'avait que t_pick/t_grade_end/t_proof_end/t_post ; ~0,5 s entre la
livraison du groupe par le flux et le début du traitement n'étaient pas
mesurés, ni la finalisation. Contrat : le flux note (t_ready, pos, bake_seq)
par prompt, le pré-bake les reprend une seule fois.
"""
from reliquary.miner import engine as eng


def test_note_and_take_ready_once():
    e = eng.MiningEngine.__new__(eng.MiningEngine)
    e._note_ready(42, 2, 7)
    t, pos, bake = e._take_ready(42)
    assert pos == 2 and bake == 7 and t > 1.7e9
    assert e._take_ready(42) is None          # consommé une seule fois
    assert e._take_ready(43) is None          # inconnu = None, jamais d'exception

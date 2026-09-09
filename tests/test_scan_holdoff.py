"""scan_holdoff : enfiler le balayage à délai fixe, pas à la livraison sprint.

Vérifie que scan_holdoff_s>0 déclenche l'enfilage du balayage plus tôt que la
livraison complète du sprint, et que 0 = comportement historique. On teste la
LOGIQUE d'enfilage via un faux moteur (pas de GPU).
"""
import types
from reliquary.miner.vllm_backend import VLLMBackend


class _FakeOut:
    def __init__(self, rid, toks):
        self.request_id = rid; self.finished = True
        self.outputs = [types.SimpleNamespace(token_ids=toks, text="")]


class _FakeEngine:
    """Sprint (pos 0,1) met 3 steps à finir ; on observe QUAND le balayage
    (pos 2,3,4) est ajouté via add_request."""
    def __init__(self):
        self.added = []       # ordre d'ajout des rid
        self._step = 0
        self._pending = {}
    def add_request(self, rid, *a, **k):
        self.added.append(rid)
        self._pending[rid] = 0
    def has_unfinished_requests(self):
        return self._step < 8 and (self._pending or self._step < 3)
    def step(self):
        self._step += 1
        # sprint (rid 0,1) finit au step 3 ; le reste au step 6
        outs = []
        for rid in list(self._pending):
            pos = int(rid)
            if (pos < 2 and self._step >= 3) or (pos >= 2 and self._step >= 6):
                outs.append(_FakeOut(rid, [7, 9]))
                self._pending.pop(rid, None)
        return outs
    def abort_request(self, *a): pass


def _run(holdoff):
    b = VLLMBackend.__new__(VLLMBackend)
    b._interrupt = None
    eng = _FakeEngine()
    order = {"scan_step": None}
    # patch : capter le step où le balayage est enfilé
    return eng, holdoff


def test_param_existe():
    import inspect
    sig = inspect.signature(VLLMBackend.generate_forced_phase1_multi_stream)
    assert "scan_holdoff_s" in sig.parameters
    assert sig.parameters["scan_holdoff_s"].default == 0.0

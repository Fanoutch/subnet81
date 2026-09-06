"""Instrumentation de la chaîne d'envoi (06/09) : submit_batch_v2 remplit un
dict ``timing`` avec les horodatages precommit/corps, dans l'ordre, sans
changer le protocole ; le chemin legacy (sans wallet) l'ignore."""
import pytest

from tests.test_precommit_port import BINDING_FIXTURE, _Recorder, _request


def _wallet():
    # Keypair via bittensor_wallet (le paquet `bittensor` complet ne s'importe
    # pas sur la dev box) — sign_precommit n'a besoin que de hotkey.sign().
    from bittensor_wallet import Keypair

    class _W:
        hotkey = Keypair.create_from_uri("//Alice")

    return _W()


@pytest.mark.asyncio
async def test_timing_rempli_dans_l_ordre():
    from reliquary.miner.submitter import submit_batch_v2

    rec = _Recorder()
    tm = {}
    async with rec.client as client:
        resp = await submit_batch_v2(
            "http://v", _request(), client=client,
            wallet=_wallet(), randomness=BINDING_FIXTURE["randomness"],
            timing=tm,
        )
    assert resp.accepted
    keys = ["t_precommit_built", "t_precommit_sent", "t_precommit_resp",
            "t_body_sent", "t_body_resp"]
    assert [k for k in keys if k in tm] == keys
    vals = [tm[k] for k in keys]
    assert vals == sorted(vals)
    # le contrat du handshake est intact
    assert [c.url.path for c in rec.calls] == ["/submit/precommit", "/submit"]


@pytest.mark.asyncio
async def test_timing_absent_ne_change_rien():
    from reliquary.miner.submitter import submit_batch_v2

    rec = _Recorder()
    async with rec.client as client:
        resp = await submit_batch_v2(
            "http://v", _request(), client=client,
            wallet=_wallet(), randomness=BINDING_FIXTURE["randomness"],
        )
    assert resp.accepted and len(rec.calls) == 2

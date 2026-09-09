"""Bascule de repo checkpoint (Discord 18/08) : le premier vrai checkpoint v4
arrive dans un NOUVEAU repo — si sa numérotation repart bas, l'ancien
déclencheur `n > local_n` ne tirait jamais. Le pull doit suivre (repo,
révision), pas seulement n."""
import asyncio
import types

from reliquary.miner.engine import maybe_pull_checkpoint


def _state(n, repo, rev):
    return types.SimpleNamespace(
        checkpoint_n=n, checkpoint_repo_id=repo, checkpoint_revision=rev)


def _run(state, local_n, local_hash):
    calls = []

    async def dl(repo, rev):
        calls.append((repo, rev)); return "/tmp/x"

    def load(path):
        return "MODEL"

    out = asyncio.run(maybe_pull_checkpoint(
        state=state, local_n=local_n, local_hash=local_hash,
        local_model="OLD", download_fn=dl, load_fn=load))
    return out, calls


def test_new_repo_low_numbering_still_pulls():
    (n, h, m), calls = _run(_state(1, "New/repo", "rev-new"), 485, "rev-485")
    assert calls == [("New/repo", "rev-new")] and m == "MODEL" and h == "rev-new"


def test_same_revision_no_pull():
    (n, h, m), calls = _run(_state(485, "Old/repo", "rev-485"), 485, "rev-485")
    assert calls == [] and m == "OLD" and n == 485


def test_normal_advance_still_pulls():
    (n, h, m), calls = _run(_state(486, "Old/repo", "rev-486"), 485, "rev-485")
    assert calls == [("Old/repo", "rev-486")] and n == 486


def test_no_published_checkpoint_noop():
    (n, h, m), calls = _run(_state(486, None, None), 485, "rev-485")
    assert calls == [] and m == "OLD"

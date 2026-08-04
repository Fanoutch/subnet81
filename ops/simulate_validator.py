"""Simulateur de validateur HORS-LIGNE — fait juger nos soumissions par le VRAI
code du validateur, sans GPU, sans réseau, sans dépenser une fenêtre.

Pourquoi : le 2026-08-03 on a découvert `bad_termination` APRÈS 21 fenêtres
gaspillées, en trois diagnostics partiels, parce que le seul retour disponible
arrivait toutes les ~45 min. Ici le verdict tombe en une seconde.

Principe : les contrôles « pas chers » du validateur sont du code PUR — ils
s'importent sans torch/vllm (vérifié). On extrait l'arbre upstream `origin/main`
et on appelle SES fonctions, pas une ré-implémentation qui pourrait dériver.

Ce qui est couvert (tout ce qui précède la preuve GPU) :
  * terminaison           admission._classify_termination   <- les 21 rejets
  * réponse mal formée    boxed_integrity.has_malformed_final_answer
  * zone / difficulté     difficulty_auction.gated_difficulty_utility
  * schéma de soumission  protocol.submission (pydantic)

Ce qui NE PEUT PAS l'être ici (chemin preuve, candidats gagnants seulement) :
  * SEED_MISMATCH / GRAIL_FAIL  -> gate GPU `validate_vllm_forced_seed_group.py`

  python scripts/simulate_validator.py              # scénarios connus
  python scripts/simulate_validator.py --in r.jsonl # rollouts réels capturés
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_REPO = "/root/subnet81/reliquary"
UPSTREAM_REF = "origin/main"
MINER = "/root/subnet81/reliquary-miner-priv"

# EOS du checkpoint 4B (ReliquaryForge/qwen3.5-4b-reliquary-v4)
EOS_4B = (248044, 248046)  # <|endoftext|>, <|im_end|>


def _extract_upstream() -> Path:
    """Arbre upstream dans un tmpdir — on juge avec SON code, pas le nôtre."""
    dest = Path(tempfile.mkdtemp(prefix="upstream_validator_"))
    proc = subprocess.run(
        ["git", "-C", UPSTREAM_REPO, "archive", UPSTREAM_REF, "reliquary"],
        capture_output=True, check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=proc.stdout, check=True)
    return dest


class _Ctx:
    """AdmissionContext minimal (duck-typing) pour _classify_termination."""

    def __init__(self, environment="opencodeinstruct", eos_ids=EOS_4B,
                 max_sequence_length=16384):
        self.environment = environment
        self.eos_token_ids = tuple(eos_ids)
        self.max_sequence_length = max_sequence_length
        self.canonical_force_ids = ()
        self.think_close_ids = ()
        self.vocab_size = None
        self.bootstrap = False


class _Rollout:
    def __init__(self, tokens, prompt_length, reward=0.0,
                 env_name="opencodeinstruct", forced=False, force_span=None):
        self.reward = reward
        self.env_name = env_name
        self.commit = {
            "tokens": list(tokens),
            "rollout": {
                "prompt_length": prompt_length,
                "completion_length": len(tokens) - prompt_length,
                "forced": forced,
                "force_span": force_span,
            },
        }


def judge_group(rollouts, *, rewards=None, environment="opencodeinstruct",
                eos_ids=EOS_4B, up=None):
    """Retourne (verdict, détails) — le verdict du VRAI code validateur."""
    sys.path.insert(0, str(up))
    from reliquary.validator.admission import _classify_termination
    from reliquary.validator.difficulty_auction import gated_difficulty_utility

    ctx = _Ctx(environment=environment, eos_ids=eos_ids)
    details = []
    for i, r in enumerate(rollouts):
        kind = _classify_termination(r, ctx)
        details.append((i, kind))
        if kind not in ("ok", "truncated"):
            return f"REJET:{kind}", details

    n_trunc = sum(1 for _, k in details if k == "truncated")
    # v3 : 3 tronqués tolérés en code, 1 ailleurs
    allowed = 3 if environment == "opencodeinstruct" else 1
    if n_trunc > allowed:
        return "REJET:bad_termination(trop de tronqués)", details

    if rewards is not None:
        util = gated_difficulty_utility(rewards, sigma_min=0.43)
        if util <= 0.0:
            return "REJET:out_of_zone", details + [("zone", f"utilité={util}")]
        details.append(("zone", f"utilité={util:.4f} → payable"))
    return "ACCEPTÉ (contrôles pas-chers)", details


def _mk(prompt_len, completion, reward=0.0):
    return _Rollout([1] * prompt_len + list(completion), prompt_len, reward)


def _load_our_fix():
    """Charge NOTRE truncate_at_first_eos, puis purge le paquet.

    Les deux arbres exposent un paquet ``reliquary`` : celui du mineur (qui n'a
    pas ``validator/admission.py``) masquerait celui d'upstream. On importe donc
    notre fonction d'abord, puis on retire le paquet de ``sys.modules`` et du
    ``sys.path`` pour laisser la place au code du validateur.
    """
    sys.path.insert(0, MINER)
    try:
        from reliquary.miner.engine import truncate_at_first_eos as fn
    finally:
        for mod in [m for m in sys.modules if m.split(".")[0] == "reliquary"]:
            del sys.modules[mod]
        sys.path.remove(MINER)
    return fn


_TRUNCATE = _load_our_fix()


def _fix(completion):
    """Sortie EFFECTIVE de notre correctif (pas une description de celui-ci).

    C'est ce qui ferme la boucle : le validateur juge ce que le mineur produit
    vraiment, donc une régression du fix casse ce simulateur.
    """
    return _TRUNCATE(completion, EOS_4B)


def scenarios():
    """Reproduit les modes de panne observés + le comportement corrigé."""
    P = 10
    return [
        ("🔴 21 rejets réels : EOS RETIRÉ par vLLM (aucun EOS)",
         [_mk(P, [5, 6, 7]) for _ in range(8)], None),
        ("🔴 ignore_eos : EOS au MILIEU puis suite",
         [_mk(P, [5, 248046, 7, 8]) for _ in range(8)], None),
        ("🔴 DEUX EOS (endoftext au milieu + im_end à la fin)",
         [_mk(P, [5, 248044, 7, 248046]) for _ in range(8)], None),
        # Boucle fermée : on ne DÉCRIT pas la sortie du fix, on l'EXÉCUTE.
        # Les entrées cassées ci-dessus passent dans NOTRE truncate_at_first_eos
        # et c'est son résultat réel qui est jugé par le validateur.
        ("🟢 APRÈS FIX : notre truncate_at_first_eos sur l'entrée « EOS au milieu »",
         [_mk(P, _fix([5, 248046, 7, 8])) for _ in range(8)], None),
        ("🟢 APRÈS FIX : notre truncate_at_first_eos sur l'entrée « deux EOS »",
         [_mk(P, _fix([5, 248044, 7, 248046])) for _ in range(8)], None),
        ("🟢 APRÈS FIX + groupe payable (k=3/8, σ=0.48)",
         [_mk(P, _fix([5, 248046, 7, 8]), reward=1.0 if i < 3 else 0.0)
          for i in range(8)],
         [1.0] * 3 + [0.0] * 5),
        ("🟡 groupe sain mais unanime (σ=0) → out_of_zone attendu",
         [_mk(P, [5, 248046], reward=0.0) for _ in range(8)], [0.0] * 8),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path",
                    help="JSONL de rollouts réels: {tokens, prompt_length, reward}")
    ap.add_argument("--env", default="opencodeinstruct")
    args = ap.parse_args()

    up = _extract_upstream()
    rev = subprocess.run(
        ["git", "-C", UPSTREAM_REPO, "rev-parse", "--short", UPSTREAM_REF],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"=== Simulateur validateur — code upstream {UPSTREAM_REF} @ {rev} ===\n")

    if args.in_path:
        rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
        rollouts = [
            _Rollout(r["tokens"], r["prompt_length"], r.get("reward", 0.0))
            for r in rows
        ]
        rewards = [r.get("reward", 0.0) for r in rows] or None
        verdict, details = judge_group(rollouts, rewards=rewards,
                                       environment=args.env, up=up)
        print(f"{len(rollouts)} rollouts réels → {verdict}")
        for i, kind in details:
            print(f"   rollout {i}: {kind}")
        raise SystemExit(0 if verdict.startswith("ACCEPTÉ") else 1)

    failed = 0
    for label, rollouts, rewards in scenarios():
        verdict, details = judge_group(rollouts, rewards=rewards,
                                       environment=args.env, up=up)
        kinds = {k for _, k in details if isinstance(k, str)}
        print(f"{label}\n    → {verdict}   (classes: {sorted(kinds)})")
        expect_ok = label.startswith("🟢")
        expect_zone = label.startswith("🟡")
        got_ok = verdict.startswith("ACCEPTÉ")
        if expect_ok and not got_ok:
            print("      ❌ ATTENDU: accepté"); failed += 1
        elif expect_zone and verdict != "REJET:out_of_zone":
            print("      ❌ ATTENDU: out_of_zone"); failed += 1
        elif label.startswith("🔴") and got_ok:
            print("      ❌ ATTENDU: rejet"); failed += 1
        print()

    print("=" * 62)
    if failed:
        print(f"❌ {failed} scénario(s) hors attente — le simulateur ou le fix dérive")
    else:
        print("✅ Tous les scénarios se comportent comme attendu.\n"
              "   Les 3 modes de panne des 21 rejets sont bien REJETÉS,\n"
              "   et la sortie du fix (troncature) est bien ACCEPTÉE\n"
              "   — jugé par le code du validateur lui-même.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

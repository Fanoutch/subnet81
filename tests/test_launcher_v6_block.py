"""Bloc launcher v6 + watchdog + restart : tests qui PARSENT/ÉVALUENT les
scripts bash sans jamais lancer le mineur.

Invariant central : sous ``RELIQUARY_PROTOCOL_VERSION=5`` (défaut) chaque
valeur exportée par ``ops/launch_miner_v4.sh`` reste STRICTEMENT celle
d'aujourd'hui ; le bloc v6 n'ajoute/ne change des valeurs que sous ``=6``.

Méthode d'évaluation : on exécute le préfixe du launcher (tout ce qui précède
``CHECKPOINT=``, donc AVANT la garde /health et l'exec du mineur) dans un
``env -i`` avec ``nvidia-smi``/``curl``/``sleep`` stubbés, puis ``export -p``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops" / "launch_miner_v4.sh"
WATCHDOG = ROOT / "ops" / "watchdog.sh"
RESTART = ROOT / "ops" / "restart_miner.sh"

# Valeurs v5 figées le 08/09 (baseline `export -p` du launcher AVANT le port v6).
V5_FROZEN = {
    "RELIQUARY_PROTOCOL_VERSION": "5",
    "RELIQUARY_FIRE_CURFEW_S": "27",
    "RELIQUARY_PREFLIP_GUARD_S": "50",
    "RELIQUARY_LATE_BAKE_FROM": "35",
    "RELIQUARY_LATE_BAKE_CAP": "1200",
    "RELIQUARY_LOCAL_TOKEN_AUTH": "0",
    "RELIQUARY_GRADE_TIMEOUT_S": "1.0",
    "RELIQUARY_VOLUME_MU": "0",
    "RELIQUARY_DRAND_MIN_HEADROOM_S": "1.0",
    "RELIQUARY_MAX_INFLIGHT_FIRES": "3",
    "RELIQUARY_CHECKPOINT_PREFETCH": "1",
    "RELIQUARY_COOLDOWN_POLL_S": "20",
}
# Variables qui n'existent PAS sous v5 (le bloc v6 ne doit pas fuir).
V6_ONLY = {
    "RELIQUARY_V6_FILL_CUTOFF_MARGIN_S",
    "RELIQUARY_STATE_RETRY_MAX_S",
    "WATCHDOG_WEDGE_S",
}
V6_EXPECTED = {
    "RELIQUARY_PROTOCOL_VERSION": "6",
    "RELIQUARY_FIRE_CURFEW_S": "0",
    "RELIQUARY_LATE_BAKE_FROM": "999999",
    "RELIQUARY_PREFLIP_GUARD_S": "999999",
    "RELIQUARY_LATE_BAKE_CAP": "1200",
    "RELIQUARY_V6_FILL_CUTOFF_MARGIN_S": "40",
    "RELIQUARY_STATE_RETRY_MAX_S": "0.25",
    "RELIQUARY_LOCAL_TOKEN_AUTH": "1",
    "RELIQUARY_GRADE_TIMEOUT_S": "5.0",
    "WATCHDOG_WEDGE_S": "2700",
    "RELIQUARY_VOLUME_MU": "0",  # candidat A/B, pas au port
}

# Contrat v5 tel que publié par /health (image 84dcc57), sha256 canonique
# 19e98f5a… ; copie de scratchpad/v6port/contract/health_live.json.
CONTRACT_V5 = {
    "profile_id": "qwen3-4b-base-dapo-reasoning-v5",
    "model_id": "Qwen/Qwen3-4B-Base",
    "model_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
    "protocol_version": 5,
    "prompt_encoding": "raw",
    "throughput_tiebreak": {"token_cap": 8192, "bucket_tokens_per_round": 50},
    "collection_seconds": 100,
    "upload_grace_seconds": 33,
    "sampling": {"rollouts": 16, "temperature": 1.0, "top_p": 1.0,
                 "top_k": 0, "do_sample": False},
    "environments": {
        "openmathinstruct": {
            "max_new_tokens": 8192, "answer_format": "boxed", "bft": None,
            "prompt_template": {
                "id": "openmathinstruct-step-by-step-v1",
                "renderer": "dollar-substitution-v1",
                "template": "Solve the following math problem step by step."
                            "\n\n$problem\n\nPut your final answer within \\boxed{}.",
                "sha256": "7f2343051fdd2a4179ddab8202cdfd765438aa3b31022c4ef8a06a5f502899c4",
            },
        },
        "opencodeinstruct": {
            "max_new_tokens": 8192, "answer_format": None, "bft": None,
            "prompt_template": {
                "id": "opencodeinstruct-step-by-step-v1",
                "renderer": "dollar-substitution-v1",
                "template": "Solve the following programming problem step by step."
                            "\n\n$problem$contract\n\nAfter your reasoning, provide "
                            "the final implementation in the last fenced Python code block.",
                "sha256": "47f2d9e16e786ae0245c85632934e08c640be8cf4b959973e9aba5f296a63380",
            },
        },
    },
}
SHA_V5 = "19e98f5a3ddac1980efe66fd80db1ec0f8db87a5e60934efd5d0e8985435eadd"
SHA_V6 = "1696eef2a8ff52284842f2253d6f699b50bc657dc93b20fc61a257db7d449385"


# ── outillage ────────────────────────────────────────────────────────────────
def _stubs(tmp_path: Path) -> Path:
    d = tmp_path / "stubs"
    d.mkdir(exist_ok=True)
    for name, body in (("nvidia-smi", "echo 0"), ("curl", "echo 200"),
                       ("sleep", "exit 0")):
        p = d / name
        p.write_text(f"#!/bin/bash\n{body}\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return d


def launcher_exports(tmp_path: Path, **env: str) -> dict[str, str]:
    """Évalue le launcher jusqu'à ``CHECKPOINT=`` et renvoie ses exports."""
    text = LAUNCHER.read_text()
    prefix = text.split("\nCHECKPOINT=", 1)[0]
    script = tmp_path / "prefix.sh"
    script.write_text(prefix + '\nexport -p | sed -n "s/^declare -x //p"\n')
    stubs = _stubs(tmp_path)
    full_env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": "/root", **env}
    out = subprocess.run(["env", "-i", *[f"{k}={v}" for k, v in full_env.items()],
                          "bash", str(script)], capture_output=True, text=True,
                         check=True)
    exports: dict[str, str] = {}
    for line in out.stdout.splitlines():
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)="(.*)"$', line)
        if m:
            exports[m.group(1)] = m.group(2).replace('\\"', '"')
    return exports


def guard_python() -> str:
    """Le heredoc python de la garde /health, extrait tel quel du launcher."""
    text = LAUNCHER.read_text()
    m = re.search(r"python - <<'EOF'.*?\n(.*?)\nEOF\n", text, re.S)
    assert m, "heredoc de la garde introuvable"
    return m.group(1)


def run_guard(tmp_path: Path, health: dict | None, *, version: str = "5"):
    """Exécute la garde avec urlopen stubbé (aucun réseau)."""
    fixture = tmp_path / "health.json"
    fixture.write_text(json.dumps(health if health is not None else {}))
    runner = tmp_path / "runner.py"
    runner.write_text(textwrap.dedent(f"""
        import io, urllib.request
        _data = open({str(fixture)!r}, "rb").read()
        urllib.request.urlopen = lambda *a, **k: io.BytesIO(_data)
        exec(compile(open({str(tmp_path / 'guard.py')!r}).read(), "guard", "exec"))
    """))
    (tmp_path / "guard.py").write_text(guard_python())
    env = {**os.environ, "PYTHONPATH": str(ROOT),
           "RELIQUARY_PROTOCOL_VERSION": version}
    return subprocess.run([sys.executable, str(runner)], capture_output=True,
                          text=True, env=env, cwd=str(ROOT))


# ── syntaxe ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("script", [LAUNCHER, WATCHDOG, RESTART])
def test_bash_syntax(script: Path):
    subprocess.run(["bash", "-n", str(script)], check=True)


# ── launcher : v5 inchangé ───────────────────────────────────────────────────
def test_launcher_v5_defaults_inchanges_texte():
    text = LAUNCHER.read_text()
    for pat in (r"RELIQUARY_FIRE_CURFEW_S:-27\}", r"RELIQUARY_PREFLIP_GUARD_S:-50\}",
                r"RELIQUARY_LATE_BAKE_FROM:-35\}", r"RELIQUARY_LOCAL_TOKEN_AUTH:-0\}",
                r"RELIQUARY_GRADE_TIMEOUT_S:-1\.0\}", r"RELIQUARY_PROTOCOL_VERSION:-5\}"):
        assert re.search(pat, text), pat


def test_launcher_v5_defaults_inchanges(tmp_path: Path):
    ex = launcher_exports(tmp_path)
    for k, v in V5_FROZEN.items():
        assert ex.get(k) == v, (k, ex.get(k))
    assert not (V6_ONLY & ex.keys()), "le bloc v6 fuit sous v5"


def test_launcher_v6_block_present():
    assert re.search(r'^if \[ "\$\{RELIQUARY_PROTOCOL_VERSION\}" = "6" \]; then',
                     LAUNCHER.read_text(), re.M)


def test_launcher_v6_block_values(tmp_path: Path):
    ex = launcher_exports(tmp_path, RELIQUARY_PROTOCOL_VERSION="6")
    for k, v in V6_EXPECTED.items():
        assert ex.get(k) == v, (k, ex.get(k))
    # Ce que v6 ne touche pas
    for k in ("RELIQUARY_DRAND_MIN_HEADROOM_S", "RELIQUARY_MAX_INFLIGHT_FIRES",
              "RELIQUARY_CHECKPOINT_PREFETCH", "RELIQUARY_COOLDOWN_POLL_S"):
        assert ex[k] == V5_FROZEN[k]


def test_launcher_v6_values_surchargeables(tmp_path: Path):
    ex = launcher_exports(tmp_path, RELIQUARY_PROTOCOL_VERSION="6",
                          RELIQUARY_FIRE_CURFEW_S="12", RELIQUARY_GRADE_TIMEOUT_S="3.0",
                          RELIQUARY_LOCAL_TOKEN_AUTH="0", WATCHDOG_WEDGE_S="3600",
                          RELIQUARY_STATE_RETRY_MAX_S="0.5")
    assert ex["RELIQUARY_FIRE_CURFEW_S"] == "12"
    assert ex["RELIQUARY_GRADE_TIMEOUT_S"] == "3.0"
    assert ex["RELIQUARY_LOCAL_TOKEN_AUTH"] == "0"
    assert ex["WATCHDOG_WEDGE_S"] == "3600"
    assert ex["RELIQUARY_STATE_RETRY_MAX_S"] == "0.5"


def test_launcher_v5_surcharge_utilisateur_toujours_respectee(tmp_path: Path):
    ex = launcher_exports(tmp_path, RELIQUARY_FIRE_CURFEW_S="12")
    assert ex["RELIQUARY_FIRE_CURFEW_S"] == "12"
    assert ex["RELIQUARY_PROTOCOL_VERSION"] == "5"


# ── garde /health : sha256 du contrat + templates ────────────────────────────
def test_garde_v5_live_fixture_passe(tmp_path: Path):
    r = run_guard(tmp_path, {"protocol_version": 5,
                             "generation_profile_id": CONTRACT_V5["profile_id"],
                             "generation_contract": CONTRACT_V5})
    assert r.returncode == 0, r.stderr
    assert "parite OK" in r.stdout


def test_garde_sans_contrat_reste_inerte(tmp_path: Path):
    r = run_guard(tmp_path, {"protocol_version": 5,
                             "generation_profile_id": CONTRACT_V5["profile_id"]})
    assert r.returncode == 0, r.stderr


def test_garde_contrat_modifie_avertit_sans_aborter(tmp_path: Path):
    """Revue item 3 : le validateur n'impose que protocole+profil ; un contrat
    retouché = AVERTISSEMENT (les deux sha), jamais un abort en boucle."""
    gc = json.loads(json.dumps(CONTRACT_V5))
    gc["collection_seconds"] = 1800
    r = run_guard(tmp_path, {"protocol_version": 5,
                             "generation_profile_id": gc["profile_id"],
                             "generation_contract": gc})
    assert r.returncode == 0, r.stderr
    assert "[garde] AVERTISSEMENT contrat" in r.stdout
    assert SHA_V5[:12] in r.stdout
    live_sha = hashlib.sha256(json.dumps(gc, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    assert live_sha[:12] in r.stdout
    assert "parite OK" in r.stdout


def test_garde_contrat_modifie_nomme_les_cles_connues(tmp_path: Path):
    gc = json.loads(json.dumps(CONTRACT_V5))
    gc["sampling"]["top_k"] = 20
    gc["model_revision"] = "deadbeef"
    r = run_guard(tmp_path, {"protocol_version": 5,
                             "generation_profile_id": gc["profile_id"],
                             "generation_contract": gc})
    assert r.returncode == 0, r.stderr
    assert "[garde] AVERTISSEMENT contrat" in r.stdout
    assert "sampling.top_k" in r.stdout and "model_revision" in r.stdout


def test_garde_detecte_template_modifie(tmp_path: Path):
    gc = json.loads(json.dumps(CONTRACT_V5))
    gc["environments"]["opencodeinstruct"]["prompt_template"]["sha256"] = "0" * 64
    r = run_guard(tmp_path, {"protocol_version": 5,
                             "generation_profile_id": gc["profile_id"],
                             "generation_contract": gc})
    assert r.returncode != 0
    assert "template opencodeinstruct" in r.stderr
    assert "contrat: sha" not in r.stderr   # le sha du contrat n'aborte plus


def test_garde_connait_les_deux_sha():
    src = guard_python()
    assert SHA_V5 in src and SHA_V6 in src


# ── watchdog ─────────────────────────────────────────────────────────────────
def _wedge_snippet() -> str:
    """Lignes du watchdog de « # v1 — wedge » à la fin de la boucle."""
    text = WATCHDOG.read_text()
    start = text.index("# v1 — wedge")
    end = text.index("\ndone", start)
    return text[start:end]


def test_watchdog_heartbeat_pattern_and_threshold():
    text = WATCHDOG.read_text()
    assert 'grep -a "heartbeat window=" | tail -1' in text   # DERNIÈRE ligne heartbeat parsée
    assert re.search(r'\[ "\$age" -gt "\$\{WATCHDOG_WEDGE_S:-900\}" \]', text)
    # warm-up post-restart intouché
    assert '[ "$since" -lt 900 ] && continue' in text
    assert not re.search(r'\[ "\$age" -gt 900 \]', text)


def _run_wedge(tmp_path: Path, log_lines: list[str], **env: str) -> str:
    log = tmp_path / "miner.log"
    log.write_text("\n".join(log_lines) + "\n")
    script = tmp_path / "wedge.sh"
    script.write_text("restart_miner() { echo \"RESTART:$1\"; }\n"
                      f"LOG={log}\nsince=5000\n" + _wedge_snippet() + "\n")
    stubs = _stubs(tmp_path)
    r = subprocess.run(["env", "-i", f"PATH={stubs}:/usr/bin:/bin", "LC_ALL=C",
                        *[f"{k}={v}" for k, v in env.items()], "bash", str(script)],
                       capture_output=True, text=True)
    return r.stdout


def _ts(age_s: int) -> str:
    out = subprocess.run(["date", "-u", "-d", f"-{age_s} seconds",
                          "+%Y-%m-%d %H:%M:%S"], capture_output=True, text=True,
                         check=True).stdout.strip()
    return out


def _hb(age_s: int, state="200", phase="collecting", quota="5/32") -> str:
    return (f"{_ts(age_s)},123 | INFO | heartbeat window=51000 state={state} "
            f"phase={phase} quota={quota} age=1s")


def test_watchdog_heartbeat_collecting_non_plein_nest_pas_une_vie(tmp_path: Path):
    """Revue item 10 : vLLM pendu avec une boucle /state vivante → restart."""
    assert "RESTART" in _run_wedge(tmp_path, [_hb(20 * 60)])
    assert "RESTART" in _run_wedge(tmp_path, [_hb(20 * 60)], WATCHDOG_WEDGE_S="2700")
    # même un heartbeat RÉCENT en collecting non plein n'est pas une vie
    assert "RESTART" in _run_wedge(tmp_path, [_hb(30)], WATCHDOG_WEDGE_S="2700")


def test_watchdog_heartbeat_repos_legitime_est_une_vie(tmp_path: Path):
    for hb in (_hb(20 * 60, quota="32/32"), _hb(20 * 60, state="503"),
               _hb(20 * 60, phase="draining")):
        assert "RESTART" not in _run_wedge(tmp_path, [hb], WATCHDOG_WEDGE_S="2700"), hb
        assert "RESTART" in _run_wedge(tmp_path, [hb]), hb   # défaut 900 s : trop vieux


def test_watchdog_heartbeat_collecting_avec_stream_fire_recent(tmp_path: Path):
    lines = [f"{_ts(600)},000 | INFO | stream_fire: groupe 1/8 prêt à 3.5s", _hb(30)]
    assert "RESTART" not in _run_wedge(tmp_path, lines, WATCHDOG_WEDGE_S="2700")
    lines = [f"{_ts(3000)},000 | INFO | stream_fire: groupe 1/8 prêt à 3.5s", _hb(30)]
    assert "RESTART" in _run_wedge(tmp_path, lines, WATCHDOG_WEDGE_S="2700")


def test_watchdog_seule_la_derniere_ligne_heartbeat_compte(tmp_path: Path):
    lines = [_hb(20 * 60, quota="32/32"), _hb(60)]   # repos puis reprise collecting
    assert "RESTART" in _run_wedge(tmp_path, lines, WATCHDOG_WEDGE_S="2700")


def test_watchdog_v5_stream_fire_inchange(tmp_path: Path):
    ok = f"{_ts(60)},000 | INFO | stream_fire: groupe 1/8 prêt à 3.5s"
    old = f"{_ts(20 * 60)},000 | INFO | stream_fire: groupe 1/8 prêt à 3.5s"
    assert "RESTART" not in _run_wedge(tmp_path, [ok])
    assert "RESTART" in _run_wedge(tmp_path, [old])


# ── restart_miner : propagation de WATCHDOG_WEDGE_S ──────────────────────────
def _wedge_block_restart() -> str:
    text = RESTART.read_text()
    m = re.search(r"# --- wedge v6 ---\n(.*?)# --- fin wedge ---", text, re.S)
    assert m, "bloc wedge absent de restart_miner.sh"
    return m.group(1)


def _eval_restart_wedge(tmp_path: Path, launcher_default: str, **env: str) -> str:
    fake = tmp_path / "launch.sh"
    fake.write_text("export RELIQUARY_PROTOCOL_VERSION=${RELIQUARY_PROTOCOL_VERSION:-%s}\n"
                    % launcher_default)
    script = tmp_path / "r.sh"
    script.write_text(f"LAUNCHER={fake}\n" + _wedge_block_restart()
                      + '\necho "WEDGE=$WATCHDOG_WEDGE_S"\n')
    r = subprocess.run(["env", "-i", "PATH=/usr/bin:/bin",
                        *[f"{k}={v}" for k, v in env.items()], "bash", str(script)],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()[-1]


def test_restart_propage_wedge(tmp_path: Path):
    assert _eval_restart_wedge(tmp_path, "5") == "WEDGE=900"
    assert _eval_restart_wedge(tmp_path, "6") == "WEDGE=2700"
    assert _eval_restart_wedge(tmp_path, "5", RELIQUARY_PROTOCOL_VERSION="6") == "WEDGE=2700"
    assert _eval_restart_wedge(tmp_path, "6", WATCHDOG_WEDGE_S="1234") == "WEDGE=1234"
    text = RESTART.read_text()
    assert re.search(r'watchdog81 "WATCHDOG_WEDGE_S=\$WATCHDOG_WEDGE_S bash /workspace/watchdog.sh',
                     text)


# ── fichier /workspace/.protocol_version (revue item 4) ──────────────────────
def _protocol_block_restart() -> str:
    text = RESTART.read_text()
    m = re.search(r"# --- protocol v6 ---\n(.*?)# --- fin protocol ---", text, re.S)
    assert m, "bloc protocol absent de restart_miner.sh"
    return m.group(1)


def _eval_restart_protocol(tmp_path: Path, **env: str) -> tuple[str, str]:
    pv = tmp_path / ".protocol_version"
    script = tmp_path / "p.sh"
    script.write_text(_protocol_block_restart()
                      + '\necho "PV=${RELIQUARY_PROTOCOL_VERSION:-}"\n')
    r = subprocess.run(["env", "-i", "PATH=/usr/bin:/bin",
                        f"RELIQUARY_PROTOCOL_VERSION_FILE={pv}",
                        *[f"{k}={v}" for k, v in env.items()], "bash", str(script)],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()[-1], (pv.read_text().strip() if pv.exists() else "")


def test_restart_ecrit_et_relit_protocol_version(tmp_path: Path):
    assert _eval_restart_protocol(tmp_path) == ("PV=", "")           # rien → rien
    assert _eval_restart_protocol(tmp_path, RELIQUARY_PROTOCOL_VERSION="6") == ("PV=6", "6")
    assert _eval_restart_protocol(tmp_path) == ("PV=6", "6")         # relu sans env
    assert _eval_restart_protocol(tmp_path, RELIQUARY_PROTOCOL_VERSION="5") == ("PV=5", "5")
    assert "/workspace/.protocol_version" in RESTART.read_text()


def test_launcher_lit_protocol_version_fichier_en_repli(tmp_path: Path):
    pv = tmp_path / ".protocol_version"
    pv.write_text("6\n")
    ex = launcher_exports(tmp_path, RELIQUARY_PROTOCOL_VERSION_FILE=str(pv))
    assert ex["RELIQUARY_PROTOCOL_VERSION"] == "6"
    assert ex["RELIQUARY_FIRE_CURFEW_S"] == "0" and ex["WATCHDOG_WEDGE_S"] == "2700"
    ex = launcher_exports(tmp_path, RELIQUARY_PROTOCOL_VERSION_FILE=str(pv),
                          RELIQUARY_PROTOCOL_VERSION="5")             # l'env gagne
    assert ex["RELIQUARY_PROTOCOL_VERSION"] == "5" and ex["RELIQUARY_FIRE_CURFEW_S"] == "27"
    assert "WATCHDOG_WEDGE_S" not in ex
    assert "RELIQUARY_PROTOCOL_VERSION_FILE" not in launcher_exports(tmp_path)
    assert "/workspace/.protocol_version" in LAUNCHER.read_text()

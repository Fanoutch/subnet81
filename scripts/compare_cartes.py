#!/usr/bin/env python3
"""Comparaison étape par étape de deux ères du mineur V1 (ex. H200 config H
contre H100) — mêmes définitions que les outils de la H200 (restartH_measure,
bake_speed, chain), appliquées aux deux côtés.

Usage :
  python3 scripts/compare_cartes.py \
    --era "H200 H:46300:46330:<submits[.gz]>:<verdicts[.gz]>:<miner.log>[,<log2>]" \
    --era "H100:46396:99999:data/h100/submits_v4.jsonl:data/h100/verdicts_v4.jsonl:/tmp/h100_miner.log"

Lecture : chaque ligne = médiane (p90) par ère. La 1re fenêtre après un
restart est exclue d'office (moteur froid), cf. --skip-first.
Pièges respectés : 1er tir par (fenêtre, prompt) dans l'ORDRE DU FICHIER (un
re-tir réécrit la même ligne) ; verdicts dédupliqués par merkle_root ;
« groupe N prêt à X s » est relatif au DÉBUT DU BAKE.
"""
import argparse, collections, gzip, json, re, statistics as st, datetime


def op(path):
    return gzip.open(path, "rt", errors="ignore") if path.endswith(".gz") else open(path, errors="ignore")


def jl(path):
    for l in op(path):
        try:
            yield json.loads(l)
        except Exception:
            continue


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def fmt(v, dec=2):
    if not v:
        return "—"
    return f"{st.median(v):.{dec}f} ({q(v, .9):.{dec}f}) n={len(v)}"


TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
def ts_of(l):
    m = TS.match(l)
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=datetime.timezone.utc).timestamp() if m else None


def era(spec, skip_first):
    label, lo, hi, subs, verds, logs = spec.split(":", 5)
    lo, hi = int(lo), int(hi)
    R = collections.OrderedDict()
    # ── journal : flip, début de bake, groupes prêts, trou entre bakes ──
    flip, open_off, ready_by_k, gaps, deliver = [], [], collections.defaultdict(list), [], []
    wins_log = set()
    cur = None          # (window, rang du bake DANS la fenêtre) — bake_seq est un compteur global
    last_end = None
    seen_bakes = collections.Counter()
    for lf in logs.split(","):
        for l in op(lf):
            if "flip_diag:" in l:
                m = re.search(r"window_n=(\d+).*signal_off=([\d.]+)", l)
                if m and lo <= int(m.group(1)) <= hi:
                    flip.append((int(m.group(1)), float(m.group(2))))
            elif "bake_start:" in l:
                m = re.search(r"window=(\d+) bake_seq=(\d+) .*open_off=([\d.]+)", l)
                if m:
                    w = int(m.group(1))
                    seen_bakes[w] += 1
                    bs = seen_bakes[w]
                    cur = (w, bs) if lo <= w <= hi else None
                    t = ts_of(l)
                    if cur and bs == 1:
                        open_off.append((w, float(m.group(3))))
                        wins_log.add(w)
                    elif cur and last_end and t and bs > 1:
                        gaps.append(t - last_end)
            elif "prêt à" in l and cur and cur[1] == 1:
                m = re.search(r"groupe (\d+)/(\d+) prêt à ([\d.]+)s", l)
                if m:
                    ready_by_k[int(m.group(1))].append((cur[0], float(m.group(3))))
            elif "bake terminé" in l:
                last_end = ts_of(l)
            elif "bake_diag:" in l and cur:
                m = re.search(r"livraison=([\d.]+)s attente_notations=([\d.]+)s", l)
                if m:
                    deliver.append((cur[0], float(m.group(2))))
    first_w = min(wins_log) if (wins_log and skip_first) else None
    keep = lambda w: w != first_w
    R["fenêtres (journal)"] = str(len([w for w in wins_log if keep(w)]))
    R["flip → signal (s)"] = fmt([x for w, x in flip if keep(w)])
    R["ouverture → début bake 1 (s)"] = fmt([x for w, x in open_off if keep(w)])
    _sig = dict(flip)
    R["signal flip → début bake 1 (s)"] = fmt([x - _sig[w] for w, x in open_off if keep(w) and w in _sig])
    for k in (1, 3, 5, 7, 10):
        R[f"bake 1 : groupe {k} prêt (s après début)"] = fmt([x for w, x in ready_by_k.get(k, []) if keep(w)], 1)
    R["attente des notations en fin de bake (s)"] = fmt([x for w, x in deliver if keep(w)])
    R["trou fin de bake → bake suivant (s)"] = fmt(gaps)

    # ── soumissions : 1er tir par (fenêtre, prompt), ordre du fichier ──
    first = {}
    for s in jl(subs):
        w = s.get("window_n") or 0
        if lo <= w <= hi and keep(w) and s.get("env", "opencodeinstruct") == "opencodeinstruct":
            first.setdefault((w, s.get("prompt_idx")), s)
    byw = collections.defaultdict(list)
    for (w, _), s in first.items():
        byw[w].append(s)
    C = collections.defaultdict(list)
    for w, rows in byw.items():
        rows = [s for s in rows if s.get("t_ready") and s.get("t_post") and s.get("flip_offset_s") is not None]
        if not rows:
            continue
        opn = st.median([s["t_post"] - s["flip_offset_s"] for s in rows])
        rows.sort(key=lambda s: s["t_ready"])
        bmin = min((s["bake_seq"] for s in rows if s.get("bake_seq") is not None), default=None)
        b1 = [s for s in rows if bmin is not None and s.get("bake_seq") == bmin]
        for i, s in enumerate(b1[:10]):
            grp = "13" if i < 3 else "48" if i < 8 else None
            if s.get("t_precommit_sent"):
                C["tir_all"].append(s["t_precommit_sent"] - opn)
                if grp == "13":
                    C["tir13"].append(s["t_precommit_sent"] - opn)
                    C["chaine13"].append(s["t_precommit_sent"] - s["t_ready"])
            if grp and s.get("t_pregrade_end") and s.get("t_repair_end"):
                C["bloc" + grp].append(s["t_repair_end"] - s["t_pregrade_end"])
            if grp and s.get("t_proof_start") and s.get("t_proof_only_end"):
                C["preuve" + grp].append(s["t_proof_only_end"] - s["t_proof_start"])
            if grp and s.get("proof_gate_wait_s") is not None:
                C["gate" + grp].append(s["proof_gate_wait_s"])
            if grp and s.get("t_pregrade_end"):
                C["pregrade" + grp].append(s["t_pregrade_end"] - s["t_ready"])
            if grp and s.get("eos_wait_s") is not None:
                C["eoswait" + grp].append(s["eos_wait_s"])
            if s.get("max_len") and s.get("t_ready"):
                pass
        for s in rows:
            if s.get("t_precommit_sent") and s.get("t_precommit_resp"):
                C["rtt"].append(s["t_precommit_resp"] - s["t_precommit_sent"])
            if s.get("eos_checked") is not None:
                C["verif"].append(s["eos_checked"])
        C["tirs_lt18"].append(sum(1 for s in rows if s.get("t_precommit_sent") and s["t_precommit_sent"] - opn < 18))
        C["tirs_lt25"].append(sum(1 for s in rows if s.get("t_precommit_sent") and s["t_precommit_sent"] - opn < 25))
    R["tir groupes 1-3 (s après ouverture)"] = fmt(C["tir13"])
    R["chaîne prêt → précommit, g1-3 (s)"] = fmt(C["chaine13"])
    R["prêt → notation finie, g1-3 (s)"] = fmt(C["pregrade13"])
    R["preuve seule g1-3 (s)"] = fmt(C["preuve13"])
    R["preuve seule g4-8 (s)"] = fmt(C["preuve48"])
    R["attente porte de preuve g1-3 (s)"] = fmt(C["gate13"])
    R["bloc preuve∥EOS g1-3 (s)"] = fmt(C["bloc13"])
    R["bloc preuve∥EOS g4-8 (s)"] = fmt(C["bloc48"])
    R["attente réplique EOS g1-3 (s)"] = fmt(C["eoswait13"])
    R["aller-retour précommit (s)"] = fmt(C["rtt"])
    R["rollouts vérifiés/groupe (moy.)"] = f"{st.mean(C['verif']):.2f}" if C["verif"] else "—"
    R["tirs < 18 s par fenêtre"] = fmt(C["tirs_lt18"], 1)
    R["tirs < 25 s par fenêtre"] = fmt(C["tirs_lt25"], 1)

    # ── ms/token (bake_speed) : prêt / max_len, par rang ──
    mlen = {}
    for (w, p), s in first.items():
        if s.get("max_len"):
            mlen[(w, s.get("prompt_idx"))] = int(s["max_len"])
    # rang k → prompt via le journal : on refait un passage léger
    mstok = collections.defaultdict(list)
    cur = None
    seen2 = collections.Counter()
    for lf in logs.split(","):
        for l in op(lf):
            if "bake_start:" in l:
                m = re.search(r"window=(\d+) bake_seq=(\d+)", l)
                if m:
                    seen2[int(m.group(1))] += 1
                cur = (int(m.group(1)), seen2[int(m.group(1))]) if m else None
            elif "prêt à" in l and cur and cur[1] == 1 and keep(cur[0]):
                m = re.search(r"groupe (\d+)/\d+ prêt à ([\d.]+)s .*prompt=(\d+)", l)
                if m and (cur[0], int(m.group(3))) in mlen and float(m.group(2)) > 0.5:
                    L = mlen[(cur[0], int(m.group(3)))]
                    mstok[int(m.group(1))].append(1000 * float(m.group(2)) / L)
    allms = [x for v in mstok.values() for x in v]
    R["ms/token effectif, bake 1 (tous rangs)"] = fmt(allms, 1)

    # ── verdicts : payés / fenêtre, acceptation par tranche ──
    best = {}
    for v in jl(verds):
        if v.get("merkle_root") and lo <= (v.get("window_n") or 0) <= hi and keep(v.get("window_n")):
            best[v["merkle_root"]] = v
    pay = collections.Counter(); ws = set(); rej = collections.Counter()
    for v in best.values():
        ws.add(v["window_n"])
        if v.get("rewarded") is True:
            pay[v["window_n"]] += 1
        r = str(v.get("reason") or "")
        if r not in ("accepted", "batch_filled", "submitted", "None", ""):
            rej[r] += 1
    seq = [pay[w] for w in sorted(ws)]
    R["payés / fenêtre (verdicts locaux)"] = fmt(seq, 2)
    R["fenêtres ≥ 7 payés"] = f"{sum(x >= 7 for x in seq)}/{len(seq)}" if seq else "—"
    R["rejets hors batch_filled"] = dict(rej.most_common(6)) or "—"
    return label, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", action="append", required=True)
    ap.add_argument("--no-skip-first", action="store_true")
    a = ap.parse_args()
    res = [era(e, not a.no_skip_first) for e in a.era]
    keys = list(res[0][1].keys())
    w0 = max(len(k) for k in keys) + 2
    print("médiane (p90) n=…".rjust(w0 + 20))
    print("".ljust(w0) + " | ".join(lab.ljust(34) for lab, _ in res))
    for k in keys:
        print(k.ljust(w0) + " | ".join(str(R.get(k, "—")).ljust(34) for _, R in res))


if __name__ == "__main__":
    main()

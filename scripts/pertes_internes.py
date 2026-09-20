#!/usr/bin/env python3
"""Pertes INTERNES du mineur (hors marché), par ère — H200 contre H100.

Répond à : « combien de groupes et de secondes perdons-nous par NOTRE fait ? »
  1. Entonnoir du 1er bake de chaque fenêtre (10 groupes) : jetés en local
     (hors zone, auth tokens, EOS), tirés par tranche, payés.
  2. Rejets du validateur qui ne sont PAS la file du marché (batch_filled).
  3. Décomposition de la chaîne prêt → précommit des groupes 1-3 (où partent
     les secondes APRÈS la génération : preuve, porte, finalize, attente drand…).
  4. Fenêtres à avancée de checkpoint contre les autres.
Même spec d'ère que compare_cartes.py : label:lo:hi:submits:verdicts:log[,log].
"""
import argparse, collections, gzip, json, re, statistics as st


def op(p):
    return gzip.open(p, "rt", errors="ignore") if p.endswith(".gz") else open(p, errors="ignore")


def jl(p):
    for l in op(p):
        try:
            yield json.loads(l)
        except Exception:
            pass


def med(v, d=2):
    return f"{st.median(v):.{d}f}" if v else "—"


def era(spec):
    lab, lo, hi, subs, verds, logs = spec.split(":", 5)
    lo, hi = int(lo), int(hi)
    # ── journal : prompts du 1er bake + rejets locaux + checkpoint ──
    b1 = collections.defaultdict(list)          # w -> prompts du 1er bake
    drop = {}                                    # prompt -> motif local
    ckpt = {}                                    # w -> ckpt_advanced
    openoff = {}
    seen = collections.Counter(); cur = None
    exc = 0
    for lf in logs.split(","):
        for l in op(lf):
            if "flip_diag:" in l:
                m = re.search(r"window_n=(\d+).*ckpt_advanced=(\w+)", l)
                if m: ckpt[int(m.group(1))] = m.group(2) == "True"
            elif "bake_start:" in l:
                m = re.search(r"window=(\d+) bake_seq=\d+ .*open_off=([\d.]+)", l)
                if m:
                    w = int(m.group(1)); seen[w] += 1
                    cur = w if (lo <= w <= hi and seen[w] == 1) else None
                    if cur: openoff[w] = float(m.group(2))
            elif "prêt à" in l and cur:
                m = re.search(r"prompt=(\d+)", l)
                if m and len(b1[cur]) < 10: b1[cur].append(int(m.group(1)))
            elif "pre_bake[" in l:
                m = re.search(r"pre_bake\[(\w+)\].*?prompt=(\d+)", l)
                if m and m.group(1) not in ("uncertain_kept", "shadow_token_auth"):
                    drop[int(m.group(2))] = m.group(1)
            elif "fire task exception" in l:
                exc += 1
    wins = sorted(w for w in b1 if w != min(b1))            # sans la fenêtre froide
    # ── soumissions : 1er tir (ordre du fichier) ──
    first = {}; opn_rows = collections.defaultdict(list)
    for s in jl(subs):
        w = s.get("window_n") or 0
        if lo <= w <= hi:
            first.setdefault((w, s.get("prompt_idx")), s)
            if s.get("t_post") and s.get("flip_offset_s") is not None:
                opn_rows[w].append(s["t_post"] - s["flip_offset_s"])
    opn = {w: st.median(v) for w, v in opn_rows.items()}
    # ── verdicts : dernier état par merkle ──
    best = {}
    for v in jl(verds):
        if v.get("merkle_root") and lo <= (v.get("window_n") or 0) <= hi:
            best[v["merkle_root"]] = v
    vby = {(v["window_n"], v.get("prompt_idx")): v for v in best.values()}

    F = collections.Counter(); nw = len(wins)
    chain = collections.defaultdict(list)
    for w in wins:
        for i, p in enumerate(b1[w]):
            F["générés"] += 1
            if p in drop:
                F["jeté local: " + drop[p]] += 1; continue
            s = first.get((w, p))
            if not s or not s.get("t_precommit_sent") or w not in opn:
                F["non tiré (autre)"] += 1; continue
            t = s["t_precommit_sent"] - opn[w]
            band = "tiré <18 s" if t < 18 else "tiré 18-25 s" if t < 25 else "tiré ≥25 s"
            F[band] += 1
            v = vby.get((w, p))
            if v and v.get("rewarded") is True: F[band + " → payé"] += 1
            if i < 3:
                g = lambda a, b: (s[b] - s[a]) if s.get(a) and s.get(b) else None
                for k, val in (("prêt→notation", g("t_ready", "t_pregrade_end")),
                               ("attente porte preuve", s.get("proof_gate_wait_s")),
                               ("preuve (seule)", g("t_proof_start", "t_proof_only_end")),
                               ("prêt→fin preuve∥EOS", g("t_ready", "t_repair_end")),
                               ("finalize", g("t_fin_start", "t_fin_end")),
                               ("attente marge drand", s.get("headroom_wait_s")),
                               ("fin preuve→précommit envoyé", g("t_repair_end", "t_precommit_sent")),
                               ("PRÊT→PRÉCOMMIT (total)", g("t_ready", "t_precommit_sent"))):
                    if val is not None: chain[k].append(val)
    # ── rejets hors file du marché, tous tirs ──
    rej = collections.Counter(); nf = 0
    for (w, p), s in first.items():
        if w not in wins: continue
        nf += 1
        v = vby.get((w, p))
        r = str((v or {}).get("reason") or s.get("reason") or "?")
        if r not in ("accepted", "batch_filled", "submitted", "None"):
            rej[r] += 1
    # ── checkpoint ──
    pay = collections.Counter(v["window_n"] for v in best.values() if v.get("rewarded") is True)
    ck = [w for w in wins if ckpt.get(w)]; nck = [w for w in wins if not ckpt.get(w)]
    return lab, nw, F, chain, rej, nf, exc, (ck, nck, openoff, pay)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--era", action="append", required=True)
    res = [era(e) for e in ap.parse_args().era]
    labs = [r[0] for r in res]
    print("\n1) ENTONNOIR DU 1er BAKE (groupes par fenêtre, moyenne)")
    keys = ["générés"] + sorted({k for r in res for k in r[2] if k.startswith("jeté")}) + \
           ["non tiré (autre)", "tiré <18 s", "tiré <18 s → payé", "tiré 18-25 s", "tiré 18-25 s → payé",
            "tiré ≥25 s", "tiré ≥25 s → payé"]
    print("  " + "".ljust(36) + "".join(f"{l:>14}" for l in labs))
    for k in keys:
        print("  " + k.ljust(36) + "".join(f"{r[2][k]/max(r[1],1):14.2f}" for r in res))
    print("\n2) REJETS HORS FILE DU MARCHÉ (tous tirs, par fenêtre)")
    for lab, nw, F, chain, rej, nf, exc, _ in res:
        print(f"  {lab}: {nf/nw:.1f} tirs/fen ; " + ", ".join(f"{k} {v/nw:.2f}" for k, v in rej.most_common(8))
              + f" ; exceptions d'envoi (déconnexion) {exc/nw:.2f}/fen")
    print("\n3) CHAÎNE DES GROUPES 1-3 (médiane, s)")
    ks = list(res[0][3].keys())
    print("  " + "".ljust(32) + "".join(f"{l:>14}" for l in labs))
    for k in ks:
        print("  " + k.ljust(32) + "".join(f"{med(r[3].get(k, [])):>14}" for r in res))
    print("\n4) FENÊTRES À AVANCÉE DE CHECKPOINT")
    for lab, nw, F, chain, rej, nf, exc, (ck, nck, oo, pay) in res:
        f = lambda ws: (med([oo[w] for w in ws if w in oo]), f"{st.mean([pay[w] for w in ws]):.2f}" if ws else "—")
        a, b = f(ck), f(nck)
        print(f"  {lab}: avec avancée {len(ck)} fen (ouverture→bake {a[0]} s, payés {a[1]}) | "
              f"sans {len(nck)} fen (ouverture→bake {b[0]} s, payés {b[1]})")


if __name__ == "__main__":
    main()

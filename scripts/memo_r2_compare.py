#!/usr/bin/env python3
"""Comparatif R2 : ère MÉMO DE TÊTE (fen ≥ --debut) contre la référence (cache --ref).

Par ère : payées/fen, % fenêtres payées, 1re arrivée (payées / non payées),
part de candidats content_in_cooldown / same_prompt_superseded / out_of_zone,
tête morte (aucun candidat < 9 s). Pull incrémental du cache mémo.
Usage : python3 scripts/memo_r2_compare.py --debut 41602 [--fin N]
"""
import argparse, glob, gzip, json, os, statistics as st, subprocess, sys, urllib.request
NOUS="5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q"; ENV="opencodeinstruct"
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def charger(cache, lo, hi):
    out=[]
    for f in sorted(glob.glob(f"{cache}/*.json.gz")):
        d=json.loads(gzip.decompress(open(f,"rb").read())); w=d["window_start"]
        if w<lo or w>hi: continue
        t0=(d.get("window_opened_wall_ts_by_environment") or {}).get(ENV)
        allc=(d.get("difficulty_auction") or {}).get(ENV,{}).get("candidates") or []
        sel=[c for c in allc if c.get("selected")]
        mine=[c for c in allc if c["hotkey"]==NOUS]
        rej=[e.get("reason") for e in d.get("rejected",[]) if e["hotkey"]==NOUS]
        arr=[c["precommit_arrival_ts"]-t0 for c in mine if t0 and c.get("precommit_arrival_ts")]
        out.append(dict(w=w, paid=sum(1 for e in d["batch"] if e["hotkey"]==NOUS and e["env_name"]==ENV and e.get("rewarded")),
            st=[c.get("status") for c in mine], first=min(arr) if arr else None, n=len(mine), rej=rej,
            barre=min((-c["throughput_rank"] for c in sel),default=None)))
    return out

def resume(lab, R):
    n=len(R)
    if not n: print(lab, "aucune fenêtre"); return
    st_all=[s for r in R for s in r["st"]]; nc=max(1,len(st_all))
    paidw=[r for r in R if r["paid"]]; unp=[r for r in R if not r["paid"]]
    med=lambda xs: round(st.median(xs),1) if xs else None
    print(f"{lab}: {n} fen | payées/fen {sum(r['paid'] for r in R)/n:.3f} | fen payées {100*len(paidw)/n:.0f}% | candidats/fen {len(st_all)/n:.2f}")
    print(f"   1re arrivée méd payées {med([r['first'] for r in paidw if r['first']])} / non payées {med([r['first'] for r in unp if r['first']])} | tête morte (aucune <9 s) {100*sum(1 for r in R if r['first'] is None or r['first']>=9)/n:.0f}% | barre méd {med([r['barre'] for r in R if r['barre']])}")
    print(f"   content_in_cooldown {100*st_all.count('content_in_cooldown')/nc:.1f}% | same_prompt_superseded {100*st_all.count('same_prompt_superseded')/nc:.1f}% | rejets validateur: {dict((k,[r2 for r in R for r2 in r['rej']].count(k)) for k in set(r2 for r in R for r2 in r['rej']))}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--debut",type=int,default=41602); ap.add_argument("--fin",type=int,default=0)
    ap.add_argument("--cache",default=f"{ROOT}/data/r2_cache_memo"); ap.add_argument("--ref",default=f"{ROOT}/data/r2_cache_0904"); a=ap.parse_args()
    fin=a.fin
    if not fin:
        try:
            h=json.load(urllib.request.urlopen("http://209.20.157.231:8080/health",timeout=15)); fin=int(h["active_window"])-3
        except Exception as e: print("health KO",e); fin=a.debut+60
    env=dict(os.environ)
    for line in open(f"{ROOT}/.env.r2"):
        if "=" in line: k,v=line.strip().split("=",1); env[k]=v
    env["AWS_ACCESS_KEY_ID"]=env["R2_ACCESS_KEY_ID"]; env["AWS_SECRET_ACCESS_KEY"]=env["R2_SECRET_ACCESS_KEY"]
    subprocess.run([sys.executable,f"{ROOT}/scripts/r2_pull_windows.py","--debut",str(a.debut),"--fin",str(fin),"--out",a.cache,"--jobs","16"],env=env,check=False,capture_output=True)
    resume("RÉFÉRENCE 41000-41335 (prior seul)", charger(a.ref,41000,41335))
    resume(f"MÉMO DE TÊTE {a.debut}-{fin}", charger(a.cache,a.debut,fin))
    # maturité : fenêtres des 25 dernières minutes peuvent encore changer
    print("(les ~15 dernières fenêtres peuvent encore mûrir — verdicts jusqu'à 28 min)")
main()

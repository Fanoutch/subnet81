import re, sys, collections
from datetime import datetime
def ts(s):
    try: return datetime.strptime(s[:19],"%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError: return None
cur=None; flip={}; memo=set(); P={}; bake=collections.Counter(); verd=[]; first_ok={}; rej=collections.Counter()
for line in open(sys.argv[1], errors="replace"):
    t=ts(line)
    if t is None: continue
    m=re.search(r"randomness flip \(window=(\d+)\)", line) or re.search(r"classement de tranche: .* — fenêtre (\d+)", line)
    if m:
        w=int(m.group(1))
        if w!=cur: cur=w; flip[cur]=t; bake[cur]=0
        continue
    if cur is None: continue
    m=re.search(r"vedette mémo: prompt=(\d+)", line)
    if m: memo.add(int(m.group(1))); continue
    m=re.search(r"groupe (\d+)/\d+ prêt à ([\d.]+)s .*prompt=(\d+)", line)
    if m:
        pos=int(m.group(1)); bake[cur]+=(pos==1); P[(cur,int(m.group(3)))]=dict(bake=bake[cur],pos=pos,ready=t,fate=None); continue
    m=re.search(r"pre_bake\[(\w+)\].*prompt=(\d+)", line)
    if m and (cur,int(m.group(2))) in P: P[(cur,int(m.group(2)))]["fate"]=m.group(1); continue
    m=re.search(r"submitted window=(\d+) prompt=(\d+) accepted=(\w+) reason=(\w+)", line)
    if m:
        w=int(m.group(1)); k=(w,int(m.group(2)))
        if m.group(3)=="True":
            if k in P: P[k]["fate"]="sent"
            if w not in first_ok: first_ok[w]=t
        elif k in P and P[k]["fate"] is None: P[k]["fate"]="rej:"+m.group(4)
        continue
    m=re.search(r"verdicts: (\d+) accepted(?:, (\d+) REJECTED (\{.*\}))?", line)
    if m and m.group(3):
        for r,n in re.findall(r"'(\w+)': (\d+)", m.group(3)): rej[r]+=int(n)
print(f"fenêtres depuis restart: {len(flip)} | picks mémo: {len(memo)}")
print(f"{'fen':>6} {'tête':>5} {'g1 mémo':>28} {'g2 mémo':>28} {'1er OK':>7}")
for w in sorted(flip):
    fb=[(p,e) for (ww,p),e in P.items() if ww==w and e["bake"]==1]
    cells=[]
    for pos in (1,2):
        e=[(p,e) for p,e in fb if e["pos"]==pos]
        if not e: cells.append("-".rjust(28)); continue
        p,e=e[0]; tag="M" if p in memo else "R"
        cells.append(f"{tag} {p} {e['fate'] or '…'} @{e['ready']-flip[w]:.0f}s".rjust(28))
    fo=f"{first_ok[w]-flip[w]:.0f}s" if w in first_ok else "-"
    print(f"{w:>6} {'':>5} {cells[0]} {cells[1]} {fo:>7}")
mh=[e for (w,p),e in P.items() if p in memo]
if mh:
    c=collections.Counter(e["fate"] for e in mh); print("têtes mémo — sorts:", dict(c), f"| hors zone: {100*sum(1 for e in mh if e['fate']=='out_of_zone')/len(mh):.0f}% (réf. classé 31 %)")
print("rejets de verdicts (cumul depuis restart):", dict(rej))

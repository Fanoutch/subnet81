import json, gzip, glob, statistics as st
from collections import defaultdict
US="5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q"
SP="/tmp/claude-0/-root-subnet81/953100f7-5d06-4052-b8eb-7e01b31ceb91/scratchpad/zone_era"
paid=defaultdict(lambda: defaultdict(list))   # hk -> w -> [(arr,vol)]
tries=defaultdict(lambda: defaultdict(int))   # hk -> w -> tentatives visibles (payées+rejetées)
rej_reasons=defaultdict(lambda: defaultdict(int))
nwin=0; wins=set()
for f in sorted(glob.glob(SP+"/*.json.gz")):
    d=json.loads(gzip.decompress(open(f,"rb").read()))
    w=d["window_start"]
    if w<38036: continue
    nwin+=1; wins.add(w)
    for e in d.get("batch",[]):
        if e.get("env_name")!="opencodeinstruct": continue
        hk=e.get("hotkey")
        vol=sum(r.get("completion_length",0) for r in e.get("rollouts",[]))
        paid[hk][w].append((e.get("arrival_age_seconds"),vol))
        tries[hk][w]+=1
    for e in d.get("rejected",[]):
        if e.get("env_name") and e.get("env_name")!="opencodeinstruct": continue
        hk=e.get("hotkey")
        if hk:
            tries[hk][w]+=1
            rej_reasons[hk][e.get("reason")]+=1
rank=sorted(paid.items(), key=lambda kv:-sum(len(v) for v in kv[1].values()))
print("=== ère batch-5, %d fenêtres archivées ==="%nwin)
print("%-10s %6s %8s %9s %9s %9s %8s %8s"%("hotkey","payées","pay/fen","fen.pay%","sièges/fp","arr méd","vol méd","tent/fen"))
for hk,byw in rank[:8]+[ (US,paid[US]) ]:
    if hk==US and any(h==US for h,_ in rank[:8]): break
    np_=sum(len(v) for v in byw.values())
    fp=len(byw)
    arr=[a for v in byw.values() for a,_ in v if a is not None]
    vol=[x for v in byw.values() for _,x in v]
    t=sum(tries[hk].values())
    tag=" <== NOUS" if hk==US else ""
    print("%-10s %6d %8.2f %8.0f%% %9.1f %8.1fs %8.0f %8.1f%s"%(hk[:8],np_,np_/nwin,100*fp/nwin,np_/max(fp,1),st.median(arr) if arr else -1,st.median(vol) if vol else -1,t/nwin,tag))
print()
# décomposition NOUS : fenêtres sans paiement — étions-nous dans les rejetés ?
uspay=set(paid[US].keys())
usseen=set(tries[US].keys())
print("NOUS: présents (payé ou rejeté visible) dans %d/%d fen, payés dans %d"%(len(usseen),nwin,len(uspay)))
print("rejets NOUS:", dict(rej_reasons[US]))
# le profil des leaders dans une fenêtre type: leurs arrivées payées
lead=rank[0][0]
arr_l=sorted(a for v in paid[lead][max(paid[lead])].values() for a in [v[0]] ) if False else None

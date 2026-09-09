#!/usr/bin/env python3
"""Effet des 2 fix (latence + file d'envoi concurrente) — RECALÉ le 20/08 22h.

⚠️ La comparaison avant/après du 20h30 est MORTE : la box a été reconstruite à
21h20 (reboot, /workspace effacé) et le rapatriement a écrasé une partie des
données de référence. On repart donc d'une mesure PROPRE depuis le redémarrage
post-reconstruction (21:52), avec la même configuration qu'avant la coupure.

LA métrique qui doit bouger : l'écart entre l'instant où une entrée est PRÊTE
(t_proof_end) et celui où elle PART (t_post). Avant le fix, un seul envoi
pouvait être en vol : quand un POST traînait, toute la fenêtre attendait
derrière puis partait d'un bloc (13-14 s mesurées sur 29888/29889).

Le piège est intermittent — la MÉDIANE ne le montre pas (0,8 s). Ce sont les
p75/p90 et la part d'entrées bloquées >3 s qu'il faut regarder.
"""
import json
import statistics as st
from collections import defaultdict

D = "/root/subnet81/data"
CUT = 1787262720          # 2026-08-20 21:52 UTC — redémarrage post-reconstruction
AVANT_DEBUT = 1787248291  # 17:51 — socle historique (partiellement perdu, cf. en-tête)


def bloc(rows, label):
    w = sorted(u["t_post"] - u["t_proof_end"] for u in rows
               if u.get("t_post") and u.get("t_proof_end"))
    if len(w) < 5:
        print(f"{label}: {len(w)} entrées — trop peu pour conclure")
        return
    n = len(w)
    offs = sorted(u["flip_offset_s"] for u in rows
                  if u.get("flip_offset_s") is not None)
    prem = defaultdict(lambda: 999.0)
    for u in rows:
        if u.get("flip_offset_s") is not None:
            prem[u["window_n"]] = min(prem[u["window_n"]], u["flip_offset_s"])
    nw = len({u["window_n"] for u in rows})
    print(f"{label}: {nw} fen, {n} entrées")
    print(f"   ATTENTE prête→partie  méd {w[n//2]:5.1f}s | p75 {w[3*n//4]:5.1f}s "
          f"| p90 {w[int(.9*n)]:5.1f}s | max {w[-1]:5.1f}s")
    print(f"   bloquées >3s : {100*sum(1 for x in w if x>3)/n:3.0f}%   "
          f"bloquées >8s : {100*sum(1 for x in w if x>8)/n:3.0f}%")
    print(f"   1re entrée méd +{st.median(prem.values()):.1f}s | "
          f"toutes entrées méd +{st.median(offs):.1f}s | "
          f"{n/max(nw,1):.1f} entrées/fen")


def motifs(rows, label):
    """stale_round : LE risque de la concurrence. Envoyer 3 requêtes d'un coup
    peut engorger la file du validateur (tolérance zéro sur le round drand),
    les 2e/3e arrivant avec un round périmé. Les reprises repassent en ~2 s,
    donc le coût n'est pas une entrée perdue mais des secondes d'arrivée."""
    from collections import Counter
    c = Counter(u.get("reason") for u in rows)
    nw = len({u["window_n"] for u in rows}) or 1
    print(f"   {label} — " + " | ".join(
        f"{k} {c[k]/nw:.2f}/fen" for k in ("submitted", "stale_round",
                                           "batch_filled") if c.get(k)))


def main():
    sub = [json.loads(l) for l in open(f"{D}/submits_v4.jsonl")]
    ver = {v["merkle_root"]: v
           for v in (json.loads(l) for l in open(f"{D}/verdicts_v4.jsonl"))
           if v.get("merkle_root")}
    acc = [u for u in sub if u.get("reason") == "submitted"]


    # MÛRISSEMENT (20/08) : le moniteur temps réel annonce le seal AVANT que
    # les verdicts ne soient remontés — une fenêtre affichée « 0 payée » se
    # révèle payante quelques minutes plus tard. Cette illusion m'a fait
    # recommander trois fois de suite des décisions opposées le soir du 20/08.
    # On ne compte donc QUE les fenêtres dont TOUS les envois acceptés ont un
    # verdict ; les autres sont explicitement écartées et signalées.
    # Le verdict arrive en DEUX temps : d'abord `reason: accepted` avec
    # rewarded=None et canonical_rank=None (reçu, mais le seal n'a pas tranché),
    # puis après le seal rewarded devient True/False. Compter une entrée
    # rewarded=None comme « non payée » sous-estime SYSTÉMATIQUEMENT — c'est ce
    # qui a produit des fenêtres annoncées à zéro qui payaient 2 ou 3 fois un
    # quart d'heure plus tard. La maturité exige donc que le seal ait DÉCIDÉ.
    def mures(rows):
        par_fen = defaultdict(lambda: [0, 0])
        for u in rows:
            par_fen[u["window_n"]][0] += 1
            v = ver.get(u.get("merkle_root"))
            if v is not None and v.get("rewarded") is not None:
                par_fen[u["window_n"]][1] += 1
        ok = {w for w, (a, d) in par_fen.items() if a and a == d}
        return [u for u in rows if u["window_n"] in ok], len(par_fen) - len(ok)

    avant, _ = mures([u for u in acc if AVANT_DEBUT < u.get("ts", 0) <= CUT])
    apres, en_cours = mures([u for u in acc if u.get("ts", 0) > CUT])
    if en_cours:
        print(f"({en_cours} fenêtre(s) récente(s) écartée(s) : verdicts "
              f"incomplets — NE PAS conclure sur elles)")

    # RÉFÉRENCE FIGÉE — mesurée le 20/08 entre 17h51 et 20h30 sur 14 fenêtres
    # TRANCHÉES (tous les envois avaient un verdict décidé), en egress direct,
    # avant le déploiement des 2 correctifs. Les fichiers bruts qui l'ont
    # produite ont été détruits à 21h45 par le rapatriement ; on la garde donc
    # en dur, c'est le seul point de comparaison solide qui subsiste.
    print("═══ FILE D'ENVOI — vs référence du 20/08 17h51-20h30 ═══")
    print("RÉFÉRENCE FIGÉE (14 fen tranchées, avant les 2 correctifs) :")
    print("   ATTENTE prête→partie  méd   0.8s | p75   4.9s | p90   9.9s")
    print("   bloquées >3s :  30%   bloquées >8s :  11%")
    print("   1re entrée méd +8.5s | 3.2 entrées/fen | 1.14 PAYÉES/fen")
    print("   stale_round 1.48/fen | batch_filled 2.84/fen")
    print()
    print()
    bloc(apres, "DEPUIS RECONSTRUCTION      ")

    print()
    motifs([u for u in sub if u.get("ts", 0) > CUT], "APRÈS")

    for rows, lbl in ((apres, "DEPUIS RECONSTRUCTION"),):
        nw = len({u["window_n"] for u in rows})
        if nw:
            paid = sum(1 for u in rows
                       if ver.get(u.get("merkle_root"), {}).get("rewarded"))
            print(f"   {lbl} payées : {paid} sur {nw} fen = {paid/nw:.2f}/fen")


if __name__ == "__main__":
    main()

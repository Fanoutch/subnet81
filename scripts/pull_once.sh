#!/bin/bash
# UN passage du rapatriement box -> dev, pour le cron (silencieux).
#
# Pourquoi c'est nécessaire alors que backup_miner_samples.sh sauvegarde les
# mêmes fichiers : la sauvegarde écrit dans data_backups/<date>/, tandis que
# scripts/train_prior_v50.py lit data/samples_v4.jsonl. Sans ce rapatriement,
# l'entraînement nocturne du prior travaillerait sur des données figées au
# 21/08 05h, et les candidats seraient entraînés sur du passé.
#
# La boucle d'origine tournait en surveillance et notifiait toutes les 30 min ;
# ici on ne veut que le transfert. INTERVAL n'est pas utilisé : on coupe après
# le premier passage, le cron se charge de la répétition.
timeout 240 bash /root/subnet81/ops/pull_samples_v4.sh >> /root/subnet81/data/pull.log 2>&1
exit 0

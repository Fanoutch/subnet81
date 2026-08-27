"""Préchargement du checkpoint suivant — le sortir du chemin critique.

POURQUOI. Mesuré le 27/08 sur la box de prod : à chaque avancée de checkpoint
la production s'arrête **67 s** (dernier bake avant → premier bake après),
contre 20 s de cadence normale entre deux bakes. La fenêtre qui SUIT l'avancée
tombe à **0,07 payée contre 0,63** en référence, et elle est totalement muette
une fois sur trois. À ~3,4 avancées par heure, cela vaut **~9 % du revenu**.

Décomposition de ces 67 s :

    téléchargement HF      57 s médian, jusqu'à 9 min 25 observé
    arrêt/relance vLLM     10 s
    chargement des poids    3 s
    compilation            13 s   (déjà divisée par 2 par le cache épinglé)
    capture des graphes     7 s

Le téléchargement domine — et il n'a **aucune raison** d'être là : il n'écrit
que des fichiers, il ne touche pas le GPU. Seul ``load_fn`` charge en VRAM.

CE QUI REND LE CORRECTIF SIMPLE. ``snapshot_download`` est IDEMPOTENT : si les
fichiers sont déjà dans le cache local il retourne instantanément. On observe
d'ailleurs déjà ce cas par accident dans le journal (9 fois sur 39 :
``Fetching 6 files: 100%|...| [00:00, 1708it/s]``). Il n'y a donc **rien à
changer dans le chemin critique** — une tâche de fond qui pré-télécharge
suffit, et ``maybe_pull_checkpoint`` devient gratuit.

L'AVANCE EXISTE, ET ELLE EST LARGE. Mesuré en croisant l'horodatage des commits
HF avec la détection par le mineur, sur 6 avancées :

    ckpt 709  publié 01:00:25  détecté 01:05:42  ->  317 s
    ckpt 710  publié 01:18:05  détecté 01:23:19  ->  314 s
    ckpt 711  publié 01:36:02  détecté 01:41:48  ->  346 s
    ckpt 712  publié 01:53:46  détecté 01:59:13  ->  327 s
    ckpt 1085 publié 09:57:49  détecté 09:59:31  ->  102 s
    ckpt 1086 publié 10:12:49  détecté 10:16:27  ->  218 s

Entre 100 et 350 s d'avance pour un téléchargement qui en demande 57.
⛔ La note du CLAUDE.md « HF publie 11 s avant, aucune avance à exploiter » est
FAUSSE — c'est sur elle qu'on avait classé le problème comme structurel.

DEUX PROPRIÉTÉS DE SÛRETÉ, toutes deux testées :

1. **On ne purge JAMAIS.** ``_hf_download`` appelle ``_prune_hf_revisions``
   qui efface toutes les révisions SAUF celle qu'on vient de tirer. Appelé
   depuis le préchargement, il supprimerait le checkpoint EN COURS D'USAGE.
   On appelle donc ``snapshot_download`` directement. La purge reste au seul
   endroit où elle est correcte : après le pull réel.

2. **Repli ouvert par construction.** Toute erreur est avalée et journalisée ;
   le chemin normal retélécharge comme aujourd'hui. On ne peut pas être plus
   mal qu'avant — au pire le préchargement n'a servi à rien.
"""
from __future__ import annotations

import asyncio
import logging
import os as _os

logger = logging.getLogger(__name__)

_DEFAULT_POLL_S = 30.0


def prefetch_enabled() -> bool:
    """Désactivé par défaut : sans la variable, comportement strictement
    identique à aujourd'hui (aucune tâche créée, aucun appel réseau)."""
    return _os.environ.get("RELIQUARY_CHECKPOINT_PREFETCH", "0") == "1"


def prefetch_poll_seconds() -> float:
    """Cadence de sondage. Les checkpoints sortent toutes les 15-18 min et
    l'avance mesurée est de 100-350 s : 30 s est largement assez fin, et
    l'appel (liste de commits) est léger."""
    try:
        v = float(_os.environ.get("RELIQUARY_CHECKPOINT_PREFETCH_POLL_S", ""))
        return v if v > 0 else _DEFAULT_POLL_S
    except (TypeError, ValueError):
        return _DEFAULT_POLL_S


def latest_published_revision(repo_id: str, list_commits_fn) -> str | None:
    """Révision la plus récemment publiée sur HF, ou ``None``.

    ``list_commits_fn`` est injecté pour les tests ; en production c'est
    ``HfApi().list_repo_commits``.
    """
    commits = list_commits_fn(repo_id)
    if not commits:
        return None
    return getattr(commits[0], "commit_id", None)


def choose_prefetch_target(
    repo_id: str | None,
    active_revision: str | None,
    list_commits_fn,
    already: set[str],
) -> str | None:
    """La révision à précharger, ou ``None`` s'il n'y a rien à faire.

    Rien à faire quand : pas de repo, la dernière publiée EST celle qu'on
    utilise déjà, ou on l'a déjà préchargée dans cette session.
    """
    if not repo_id:
        return None
    latest = latest_published_revision(repo_id, list_commits_fn)
    if latest is None or latest == active_revision or latest in already:
        return None
    return latest


async def prefetch_once(
    repo_id: str,
    revision: str,
    *,
    download_fn,
) -> bool:
    """Télécharge une révision. Ne purge RIEN. Ne lève jamais.

    ``download_fn(repo_id, revision)`` est synchrone et part dans un thread —
    le téléchargement ne doit pas geler la boucle d'événements.
    """
    try:
        await asyncio.to_thread(download_fn, repo_id, revision)
        logger.info(
            "préchargement du checkpoint OK: %s@%s — le pull réel sera instantané",
            repo_id, revision[:12],
        )
        return True
    except Exception:
        # Repli OUVERT : le chemin normal retéléchargera. Jamais fatal.
        logger.warning(
            "préchargement du checkpoint échoué (%s@%s) — sans conséquence, "
            "le pull normal prendra le relais", repo_id, revision[:12],
            exc_info=True,
        )
        return False


async def prefetch_loop(
    *,
    get_active,
    list_commits_fn,
    download_fn,
    sleep_fn=asyncio.sleep,
    poll_s: float | None = None,
    max_rounds: int | None = None,
) -> int:
    """Boucle de fond. Retourne le nombre de préchargements réussis.

    ``get_active()`` rend ``(repo_id, revision_active)`` — l'état courant du
    mineur. ``max_rounds`` borne la boucle pour les tests ; en production elle
    tourne jusqu'à l'annulation de la tâche.
    """
    done: set[str] = set()
    ok = 0
    rounds = 0
    delay = poll_s if poll_s is not None else prefetch_poll_seconds()
    while max_rounds is None or rounds < max_rounds:
        rounds += 1
        try:
            repo_id, active = get_active()
            target = choose_prefetch_target(repo_id, active, list_commits_fn, done)
            if target is not None:
                done.add(target)          # marqué AVANT : pas de double tir
                if await prefetch_once(repo_id, target, download_fn=download_fn):
                    ok += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            # Même une erreur de sondage ne doit pas tuer la tâche.
            logger.warning("boucle de préchargement: tour ignoré", exc_info=True)
        await sleep_fn(delay)
    return ok

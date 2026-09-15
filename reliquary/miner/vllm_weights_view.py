"""Poids chargés par vLLM = poids chargés par le validateur.

Mesuré le 15/09 (fenêtre 45969) : les snapshots du dépôt de checkpoints portent
``model.safetensors`` (réécrit à chaque checkpoint) ET des shards
``model-0000x-of-0000n.safetensors`` + ``model.safetensors.index.json`` figés
depuis le 14/09 08:31. transformers — donc le validateur, notre modèle de preuve
et la réplique — charge ``model.safetensors`` en priorité ; vLLM
(``filter_duplicate_safetensors_files``) ne garde que les fichiers de l'index,
donc les shards périmés. Nos tokens étaient tirés d'un modèle différent de celui
qui les vérifie (1er token exact 2-3/16 contre 84-97 % chez les autres mineurs).

Dès que ``model.safetensors`` coexiste avec un index, on donne à vLLM une vue du
snapshot (liens symboliques) sans l'index ni les shards. Le ``config.json`` est
le même fichier : la clé du cache torch.compile ne change pas.
Repli : ``RELIQUARY_VLLM_WEIGHTS_VIEW=0``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile

logger = logging.getLogger(__name__)

SINGLE = "model.safetensors"
INDEX = "model.safetensors.index.json"


def _default_root(env) -> str:
    return env.get("RELIQUARY_VLLM_WEIGHTS_VIEW_ROOT") or os.path.join(
        tempfile.gettempdir(), "reliquary_vllm_weights_view")


def _excluded(snapshot: str) -> set[str]:
    excluded = {INDEX}
    try:
        with open(os.path.join(snapshot, INDEX)) as fh:
            excluded.update(str(v) for v in (json.load(fh).get("weight_map") or {}).values())
    except (OSError, ValueError, AttributeError):
        pass
    for name in os.listdir(snapshot):
        if name.startswith("model-") and name.endswith(".safetensors"):
            excluded.add(name)
    return excluded


def _view_is_current(view: str, snapshot: str, wanted: dict[str, str]) -> bool:
    try:
        present = set(os.listdir(view))
    except OSError:
        return False
    if present != set(wanted):
        return False
    return all(os.path.realpath(os.path.join(view, n)) == t for n, t in wanted.items())


def _purge_dangling_views(root: str, *, keep: str) -> None:
    """Vues dont le snapshot a été purgé (lien model.safetensors mort)."""
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if path == keep or name.startswith(".") or not os.path.isdir(path):
            continue
        link = os.path.join(path, SINGLE)
        if os.path.islink(link) and not os.path.exists(link):
            shutil.rmtree(path, ignore_errors=True)


def vllm_weights_path(model_path: str, *, view_root: str | None = None,
                      env=None) -> str:
    """Chemin à passer à vLLM pour ``model_path`` (jamais d'exception)."""
    src = os.environ if env is None else env
    if src.get("RELIQUARY_VLLM_WEIGHTS_VIEW", "1") == "0":
        return model_path
    try:
        if not os.path.isdir(model_path):
            return model_path
        if not (os.path.exists(os.path.join(model_path, SINGLE))
                and os.path.exists(os.path.join(model_path, INDEX))):
            return model_path
        excluded = _excluded(model_path)
        wanted = {n: os.path.realpath(os.path.join(model_path, n))
                  for n in os.listdir(model_path) if n not in excluded}
        root = view_root or _default_root(src)
        real = os.path.realpath(model_path)
        tag = hashlib.sha256(real.encode()).hexdigest()[:12]
        view = os.path.join(root, f"{os.path.basename(real.rstrip('/'))}-{tag}")
        if _view_is_current(view, model_path, wanted):
            return view
        os.makedirs(root, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=".build-", dir=root)
        for name, target in wanted.items():
            os.symlink(target, os.path.join(tmp, name))
        if os.path.lexists(view):
            shutil.rmtree(view, ignore_errors=True)
        os.replace(tmp, view)
        _purge_dangling_views(root, keep=view)
        logger.info(
            "vLLM : snapshot %s porte %s ET des shards indexés (%d fichiers "
            "exclus) — chargement de %s via la vue %s, comme le validateur",
            model_path, SINGLE, len(excluded & set(os.listdir(model_path))),
            SINGLE, view)
        return view
    except Exception:
        logger.exception("vue des poids vLLM impossible — chargement direct de %s",
                         model_path)
        return model_path

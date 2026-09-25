"""Veto du cooldown de CONTENU par empreinte (24/09, mineur math).

Le validateur refuse (``content_in_cooldown``) un prompt dont le texte rendu
a déjà été sélectionné — cooldown de 1 000 000 fenêtres, donc définitif.
OMI-2 : ~14 M lignes pour bien moins d'énoncés distincts, 114 888 empreintes
math brûlées au 24/09. Le code a un veto par INDEX pré-calculé (2,48 M
digests) ; pour 14 M lignes on calcule l'empreinte au tirage.
"""
import hashlib

from reliquary.miner.engine import burned_digests_load, prompt_content_digest


def test_digest_matches_validator_formula():
    # validator/prompt_content.py : sha256(domaine + env + "\0" + prompt)
    h = hashlib.sha256()
    h.update(b"reliquary/prompt-content/v1\0")
    h.update(b"openmathinstruct")
    h.update(b"\0")
    h.update("Solve x.\n\nPut $\\boxed{}$ é".encode("utf-8"))
    assert prompt_content_digest(
        "openmathinstruct", "Solve x.\n\nPut $\\boxed{}$ é") == h.hexdigest()


def test_digest_is_env_scoped():
    assert prompt_content_digest("openmathinstruct", "p") != prompt_content_digest(
        "opencodeinstruct", "p")


def test_load_text_file_one_hex_per_line(tmp_path):
    f = tmp_path / "burned.txt"
    f.write_text("AA" * 32 + "\n\n" + "bb" * 32 + "\n")
    assert burned_digests_load(str(f)) == frozenset({"aa" * 32, "bb" * 32})


def test_load_missing_file_is_empty(tmp_path):
    assert burned_digests_load(str(tmp_path / "absent.txt")) == frozenset()

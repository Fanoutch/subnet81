"""Carnet de curseurs du mineur SFT (27/09).

Le validateur corpus exige des soumissions STRICTEMENT dans l'ordre du curseur
de la hotkey (``bad_cursor`` sinon), mais le prompt de chaque curseur est connu
d'avance (``walk_index``). Le carnet laisse donc générer plusieurs curseurs à
la fois tout en ne rendant au soumetteur que le prochain attendu.
"""
from sft_miner.book import CursorBook


def _admit_all(book):
    out = []
    while (c := book.next_admission()) is not None:
        out.append(c)
    return out


def test_admits_consecutive_cursors_up_to_in_flight_limit():
    book = CursorBook(start=10, max_in_flight=3, max_ahead=8)
    assert _admit_all(book) == [10, 11, 12]


def test_submits_in_cursor_order_even_if_later_finish_first():
    book = CursorBook(start=0, max_in_flight=3, max_ahead=8)
    _admit_all(book)
    book.on_generated(2, "g2")
    book.on_generated(1, "g1")
    assert book.take_submittable() is None          # 0 pas prêt : rien ne part
    book.on_generated(0, "g0")
    assert book.take_submittable() == (0, "g0")
    assert book.take_submittable() is None          # une soumission à la fois
    book.on_accepted(0)
    assert book.take_submittable() == (1, "g1")
    book.on_accepted(1)
    assert book.take_submittable() == (2, "g2")


def test_failed_prompt_is_readmitted_before_new_ones():
    book = CursorBook(start=0, max_in_flight=2, max_ahead=8)
    assert _admit_all(book) == [0, 1]
    book.on_failed(0)                                # préemption : preuve invalide
    assert book.next_admission() == 0               # régénéré en priorité
    assert book.next_admission() is None             # 2 en vol : plein


def test_admission_bounded_by_distance_to_next_submission():
    book = CursorBook(start=0, max_in_flight=10, max_ahead=3)
    assert _admit_all(book) == [0, 1, 2]
    book.on_generated(0, "g0")
    assert book.next_admission() is None             # 0 prêt mais pas soumis
    book.take_submittable()
    book.on_accepted(0)
    assert book.next_admission() == 3


def test_freed_in_flight_slot_admits_next():
    book = CursorBook(start=0, max_in_flight=2, max_ahead=8)
    _admit_all(book)
    book.on_generated(1, "g1")
    assert book.next_admission() == 2


def test_realign_forward_drops_stale_and_aborts_in_flight():
    book = CursorBook(start=0, max_in_flight=4, max_ahead=8)
    _admit_all(book)                                 # 0..3 en vol
    book.on_generated(0, "g0")
    book.on_generated(1, "g1")
    book.realign(2)                                  # le validateur est à 2
    assert sorted(book.pop_aborts()) == []           # 0 et 1 déjà finis, rien à couper
    assert book.take_submittable() is None           # 2 pas prêt
    book.realign(5)
    assert sorted(book.pop_aborts()) == [2, 3]       # en vol mais périmés
    assert book.next_admission() == 5


def test_realign_same_cursor_regenerates_it_if_missing():
    book = CursorBook(start=0, max_in_flight=2, max_ahead=8)
    _admit_all(book)
    book.on_generated(0, "g0")
    assert book.take_submittable() == (0, "g0")
    book.realign(0)                                  # refusé sans avancer
    assert book.next_admission() == 0                # à régénérer (graine neuve)


def test_stale_generation_is_ignored():
    book = CursorBook(start=0, max_in_flight=2, max_ahead=8)
    _admit_all(book)
    book.realign(3)
    book.on_generated(1, "g1")                       # arrive après le réalignement
    assert book.take_submittable() is None
    assert book.next_admission() == 3


def test_stop_halts_admission_and_submission():
    book = CursorBook(start=0, max_in_flight=2, max_ahead=8)
    book.stop("job_complete")
    assert book.stopped == "job_complete"
    assert book.next_admission() is None
    assert book.take_submittable() is None

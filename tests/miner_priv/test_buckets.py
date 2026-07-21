from unittest.mock import patch, MagicMock
from reliquary.miner.buckets import BucketIndex


def _fake_dataset():
    """Mimic qwedsacf/competition_math row schema."""
    return [
        {"problem": "p0", "level": "Level 3", "type": "Algebra"},
        {"problem": "p1", "level": "Level 5", "type": "Counting & Probability"},
        {"problem": "p2", "level": "Level 1", "type": "Algebra"},
    ]


@patch("reliquary.miner.buckets._load_raw_dataset")
def test_bucket_of_returns_canonical_key(mock_load):
    mock_load.return_value = _fake_dataset()
    idx = BucketIndex()
    assert idx.bucket_of(0) == ("Algebra", "Level 3")
    assert idx.bucket_of(1) == ("Counting & Probability", "Level 5")
    assert idx.bucket_of(2) == ("Algebra", "Level 1")


@patch("reliquary.miner.buckets._load_raw_dataset")
def test_bucket_of_handles_missing_fields(mock_load):
    mock_load.return_value = [
        {"problem": "p0"},                       # no level, no type
        {"problem": "p1", "level": "Level 2"},   # no type
    ]
    idx = BucketIndex()
    assert idx.bucket_of(0) == ("unknown", "unknown")
    assert idx.bucket_of(1) == ("unknown", "Level 2")


@patch("reliquary.miner.buckets._load_raw_dataset")
def test_bucket_of_wraps_modulo(mock_load):
    mock_load.return_value = _fake_dataset()
    idx = BucketIndex()
    # Out-of-range index wraps modulo dataset length (mirror env.get_problem)
    assert idx.bucket_of(3) == idx.bucket_of(0)
    assert idx.bucket_of(7) == idx.bucket_of(1)

def test_virtual_parquet_imports_and_constructs():
    from reliquary.environment.virtual_parquet import VirtualParquetDataset
    # Construction pure (pas de réseau tant qu'on n'appelle pas len/getitem).
    ds = VirtualParquetDataset("owner/repo", "rev123", columns=["input", "structured_cases"])
    assert ds._repo == "owner/repo" and ds._revision == "rev123"
    assert ds._columns == ["input", "structured_cases"]

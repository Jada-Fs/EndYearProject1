"""Tests for schema checks and join-key selection."""

import pandas as pd
import pytest

from smart_agri_gnn.validation import (
    ValidationError,
    choose_join_key,
    choose_target_column,
    validate_climate,
    validate_processed,
    validate_yield,
)


def test_validate_climate_requires_temperature():
    with pytest.raises(ValidationError, match="temperature"):
        validate_climate(pd.DataFrame({"country_code": ["UGA"], "year": [2023]}))


def test_validate_yield_requires_target():
    with pytest.raises(ValidationError, match="yield"):
        validate_yield(pd.DataFrame({"country_code": ["UGA"]}))


def test_choose_join_key_prefers_country_code():
    frame = pd.DataFrame({"country_code": ["UGA"], "region": ["UGA-00"]})
    assert choose_join_key(frame, frame) == "country_code"


def test_choose_target_column():
    frame = pd.DataFrame({"yield": [1.2], "temperature": [22.0]})
    assert choose_target_column(frame) == "yield"


def test_validate_processed_rejects_missing_target():
    frame = pd.DataFrame({"yield": [1.0, None], "temperature": [0.1, 0.2]})
    with pytest.raises(ValidationError, match="missing"):
        validate_processed(frame, "yield", ["temperature"])

import copy

import pytest

from server.taxonomy import load_taxonomy, validate_taxonomy


def test_real_file_valid():
    validate_taxonomy()


def test_missing_category_rejected():
    t = copy.deepcopy(load_taxonomy())
    del t["categories"]["appeasement"]
    with pytest.raises(ValueError, match="missing categories"):
        validate_taxonomy(t)


def test_extra_category_rejected():
    t = copy.deepcopy(load_taxonomy())
    t["categories"]["made_up"] = t["categories"]["appeasement"]
    with pytest.raises(ValueError, match="unknown categories"):
        validate_taxonomy(t)


def test_band_gap_rejected():
    t = copy.deepcopy(load_taxonomy())
    t["score_bands"][1]["min"] = 0.2  # breaks contiguity with band 0's max of 0.1
    with pytest.raises(ValueError, match="contiguous"):
        validate_taxonomy(t)


def test_bad_color_rejected():
    t = copy.deepcopy(load_taxonomy())
    t["score_bands"][0]["color"] = "green"
    with pytest.raises(ValueError, match="color"):
        validate_taxonomy(t)


def test_bad_threshold_rejected():
    t = copy.deepcopy(load_taxonomy())
    t["display_threshold"] = 1.5
    with pytest.raises(ValueError, match="display_threshold"):
        validate_taxonomy(t)

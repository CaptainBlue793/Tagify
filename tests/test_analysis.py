import pytest

from analysis import analyze, parse_taxonomy, score_industries
from Tagify import SAMPLE


@pytest.mark.parametrize("text", ["", " \n ", "!!! 123", "the and or"])
def test_invalid_input_is_actionable(text):
    with pytest.raises(ValueError):
        analyze(text)


def test_unmatched_text_has_no_arbitrary_labels():
    assert score_industries("a peculiar purple umbrella") == []


def test_matches_escape_punctuation_and_respect_word_boundaries():
    custom = {"Test": ["C++", "risk assessment", "art", "C++"]}
    rows = score_industries("C++ C++ risk   assessment partial", custom)
    test = next(row for row in rows if row["industry"] == "Test")
    assert test["score"] == 2
    assert test["occurrences"] == 3
    assert "art" not in test["matched_keywords"]


def test_short_input_uses_keywords():
    result = analyze("Robotics actuators control robotic motion.")
    assert result["mode"] == "keywords"
    assert len(result["topics"]) == 1


def test_topic_model_is_repeatable_and_segmented():
    first, second = analyze(SAMPLE), analyze(SAMPLE)
    assert first == second
    assert first["mode"] == "lda"
    assert first["segments"] > 1
    assert abs(sum(row["weight"] for row in first["topics"]) - 1) < 1e-8


def test_custom_taxonomy_is_validated():
    assert parse_taxonomy("Robotics: robot, actuator") == {"Robotics": ["robot", "actuator"]}
    with pytest.raises(ValueError):
        parse_taxonomy("No delimiter")


def test_oversized_input_is_rejected():
    with pytest.raises(ValueError):
        analyze("word " * 60001)

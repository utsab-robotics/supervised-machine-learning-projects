import pytest

from src.predict import predict_species


def test_setosa_prediction():
    result = predict_species([5.1, 3.5, 1.4, 0.2])
    assert result["species"] == "setosa"


def test_versicolor_prediction():
    result = predict_species([6.0, 2.9, 4.5, 1.5])
    assert result["species"] == "versicolor"


def test_virginica_prediction():
    result = predict_species([6.5, 3.0, 5.8, 2.2])
    assert result["species"] == "virginica"


def test_confidence_is_valid():
    result = predict_species([5.1, 3.5, 1.4, 0.2])
    assert 0.0 <= result["confidence"] <= 1.0


def test_probabilities_sum_to_one():
    result = predict_species([5.1, 3.5, 1.4, 0.2])
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_wrong_number_of_measurements():
    with pytest.raises(ValueError):
        predict_species([5.1, 3.5, 1.4])


def test_non_numeric_measurements():
    with pytest.raises(ValueError):
        predict_species(["hello", 3.5, 1.4, 0.2])


def test_zero_measurement():
    with pytest.raises(ValueError):
        predict_species([0, 3.5, 1.4, 0.2])
from unittest.mock import MagicMock

from numpy import iinfo, int32
from torch import randint

from ad_denoise.datasets.repeating import RepeatingDataset


def test_repeating_produces_frames():
    data = randint(0, iinfo(int32).max, size=(10, 10))
    mock_dataset = MagicMock(__getitem__=MagicMock(return_value=data))
    for frame in RepeatingDataset(mock_dataset, 4):
        assert (data == frame).all()


def test_repeating_reports_length():
    data = randint(0, iinfo(int32).max, size=(10, 10))
    mock_dataset = MagicMock(__getitem__=MagicMock(return_value=data))
    assert 42 == len(RepeatingDataset(mock_dataset, 42))


def test_repeating_iterates_for_apparent_length():
    data = randint(0, iinfo(int32).max, size=(10, 10))
    mock_dataset = MagicMock(__getitem__=MagicMock(return_value=data))
    for idx, _ in enumerate(RepeatingDataset(mock_dataset, 4)):
        ...
    assert 4 == idx

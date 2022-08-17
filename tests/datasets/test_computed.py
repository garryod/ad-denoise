from unittest.mock import MagicMock

from torch import tensor

from ad_denoise.datasets.computed import ComputedFramesDataset


def test_computed_does_computation():
    mock_dataset = MagicMock(
        __getitem__=MagicMock(return_value=(tensor([1, 2, 3]), tensor([4, 5, 6])))
    )
    dataset = ComputedFramesDataset(mock_dataset, lambda data: data[0] + data[1])
    assert (tensor([5, 7, 9]) == dataset[0]).all()


def test_computed_passes_length():
    mock_dataset = MagicMock(__len__=MagicMock(return_value=42))
    dataset = ComputedFramesDataset(mock_dataset, lambda data: None)
    assert 42 == len(dataset)

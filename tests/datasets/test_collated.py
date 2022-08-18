from unittest.mock import MagicMock

from pytest import raises
from torch import iinfo, int32, randint

from ad_denoise.datasets.collated import CrossedDatasets, ZippedDatasets


def test_zipped_produces_frames_single():
    data = randint(iinfo(int32).max, size=(10, 10, 10))
    mock_dataset = MagicMock(__getitem__=lambda idx: data[idx])
    for idx, (frame,) in enumerate(ZippedDatasets(mock_dataset)):
        assert (data[idx] == frame).all()


def test_zipped_produces_frames_multiple():
    data1 = randint(iinfo(int32).max, size=(10, 10, 10))
    data2 = randint(iinfo(int32).max, size=(10, 10, 10))
    data3 = randint(iinfo(int32).max, size=(10, 10, 10))
    mock_dataset1 = MagicMock(__getitem__=lambda idx: data1[idx])
    mock_dataset2 = MagicMock(__getitem__=lambda idx: data2[idx])
    mock_dataset3 = MagicMock(__getitem__=lambda idx: data3[idx])
    for idx, (frame1, frame2, frame3) in enumerate(
        ZippedDatasets(mock_dataset1, mock_dataset2, mock_dataset3)
    ):
        assert (data1[idx] == frame1).all()
        assert (data2[idx] == frame2).all()
        assert (data3[idx] == frame3).all()


def test_zipped_reports_length():
    mock_dataset1 = MagicMock(__len__=MagicMock(return_value=42))
    mock_dataset2 = MagicMock(__len__=MagicMock(return_value=42))
    assert 42 == len(ZippedDatasets(mock_dataset1, mock_dataset2))


def test_different_lengths_raises_on_init():
    mock_short_dataset = MagicMock(__len__=MagicMock(return_value=16))
    mock_long_dataset = MagicMock(__len__=MagicMock(return_value=64))
    with raises(ValueError):
        ZippedDatasets(mock_short_dataset, mock_long_dataset)


def test_crossed_produces_frames_single():
    data = randint(iinfo(int32).max, size=(10, 1, 1))
    mock_dataset = MagicMock(__getitem__=lambda idx: data[idx])
    for idx, ((frame,),) in enumerate(CrossedDatasets(mock_dataset)):
        assert (data[idx] == frame).all()


def test_crossed_produces_frames_multiple():
    data1 = randint(iinfo(int32).max, size=(1, 10, 10))
    data2 = randint(iinfo(int32).max, size=(2, 10, 10))
    data3 = randint(iinfo(int32).max, size=(3, 10, 10))
    mock_dataset1 = MagicMock(__getitem__=lambda idx: data1[idx])
    mock_dataset2 = MagicMock(__getitem__=lambda idx: data2[idx])
    mock_dataset3 = MagicMock(__getitem__=lambda idx: data3[idx])
    for idx, ((frame1,), (frame2,), (frame3,)) in enumerate(
        CrossedDatasets(mock_dataset1, mock_dataset2, mock_dataset3)
    ):
        assert (data1[0] == frame1).all()
        assert (data2[idx % 2] == frame2).all()
        assert (data3[(idx // 2) % 3] == frame3).all()


def test_crossed_reports_length():
    mock_dataset1 = MagicMock(__len__=MagicMock(return_value=1))
    mock_dataset2 = MagicMock(__len__=MagicMock(return_value=2))
    mock_dataset3 = MagicMock(__len__=MagicMock(return_value=3))
    assert 6 == len(CrossedDatasets(mock_dataset1, mock_dataset2, mock_dataset3))

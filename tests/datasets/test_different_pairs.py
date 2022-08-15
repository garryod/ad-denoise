from pathlib import Path
from tempfile import TemporaryDirectory

from h5py import File
from numpy import iinfo, int32, split
from numpy.random import randint
from pytest import raises
from torch import from_numpy

from ad_denoise.datasets.different_pairs import CrossedFramePairs, MatchedFramePairs


def test_matched_pairs_produces_frames():
    data = randint(iinfo(int32).max, size=(20, 10, 10))
    left_data, right_data = split(data, 2)
    with TemporaryDirectory() as tmpdir:
        left_file_path = Path(tmpdir).joinpath("left_testfile.h5")
        with File(left_file_path, "w") as left_file:
            left_file["left_dataset"] = left_data
        right_file_path = Path(tmpdir).joinpath("right_testfile.h5")
        with File(right_file_path, "w") as right_file:
            right_file["right_dataset"] = right_data
        for idx, (left_frame, right_frame) in enumerate(
            MatchedFramePairs(
                {left_file_path}, {right_file_path}, "left_dataset", "right_dataset"
            )
        ):
            assert (from_numpy(left_data[idx]) == left_frame).all()
            assert (from_numpy(right_data[idx]) == right_frame).all()


def test_matched_pairs_produces_frames_multiple():
    data = randint(iinfo(int32).max, size=(40, 10, 10))
    left_data, right_data = split(data, 2)
    left_data1, left_data2 = split(left_data, (8,))
    right_data1, right_data2 = split(right_data, (12,))
    with TemporaryDirectory() as tmpdir:
        left_file_path1 = Path(tmpdir).joinpath("left_testfile1.h5")
        with File(left_file_path1, "w") as left_file1:
            left_file1["left_dataset"] = left_data1
        left_file_path2 = Path(tmpdir).joinpath("left_testfile2.h5")
        with File(left_file_path2, "w") as left_file2:
            left_file2["left_dataset"] = left_data2
        right_file_path1 = Path(tmpdir).joinpath("right_testfile1.h5")
        with File(right_file_path1, "w") as right_file1:
            right_file1["right_dataset"] = right_data1
        right_file_path2 = Path(tmpdir).joinpath("right_testfile2.h5")
        with File(right_file_path2, "w") as right_file2:
            right_file2["right_dataset"] = right_data2
        for idx, (left_frame, right_frame) in enumerate(
            MatchedFramePairs(
                [left_file_path1, left_file_path2],
                [right_file_path1, right_file_path2],
                "left_dataset",
                "right_dataset",
            )
        ):
            assert (from_numpy(left_data[idx]) == left_frame).all()
            assert (from_numpy(right_data[idx]) == right_frame).all()


def test_matched_pairs_reports_length():
    data = randint(iinfo(int32).max, size=(40, 10, 10))
    left_data, right_data = split(data, 2)
    left_data1, left_data2 = split(left_data, (8,))
    with TemporaryDirectory() as tmpdir:
        left_file_path1 = Path(tmpdir).joinpath("left_testfile1.h5")
        with File(left_file_path1, "w") as left_file1:
            left_file1["left_dataset"] = left_data1
        left_file_path2 = Path(tmpdir).joinpath("left_testfile2.h5")
        with File(left_file_path2, "w") as left_file2:
            left_file2["left_dataset"] = left_data2
        right_file_path = Path(tmpdir).joinpath("right_testfile.h5")
        with File(right_file_path, "w") as right_file:
            right_file["right_dataset"] = right_data
        assert 20 == len(
            MatchedFramePairs(
                [left_file_path1, left_file_path2],
                {right_file_path},
                "left_dataset",
                "right_dataset",
            )
        )


def test_different_lengths_raises_on_init():
    data = randint(iinfo(int32).max, size=(40, 10, 10))
    left_data, right_data = split(data, (18,))
    with TemporaryDirectory() as tmpdir:
        left_file_path = Path(tmpdir).joinpath("left_testfile.h5")
        with File(left_file_path, "w") as left_file:
            left_file["left_dataset"] = left_data
        right_file_path = Path(tmpdir).joinpath("right_testfile.h5")
        with File(right_file_path, "w") as right_file:
            right_file["right_dataset"] = right_data
        with raises(ValueError):
            MatchedFramePairs(
                {left_file_path}, {right_file_path}, "left_dataset", "right_dataset"
            )


def test_crossed_pairs_produces_frames():
    data = randint(iinfo(int32).max, size=(20, 10, 10))
    left_data, right_data = split(data, 2)
    with TemporaryDirectory() as tmpdir:
        left_file_path = Path(tmpdir).joinpath("left_testfile.h5")
        with File(left_file_path, "w") as left_file:
            left_file["left_dataset"] = left_data
        right_file_path = Path(tmpdir).joinpath("right_testfile.h5")
        with File(right_file_path, "w") as right_file:
            right_file["right_dataset"] = right_data
        for idx, (left_frame, right_frame) in enumerate(
            CrossedFramePairs(
                {left_file_path}, {right_file_path}, "left_dataset", "right_dataset"
            )
        ):
            assert (from_numpy(left_data[idx // len(left_data)]) == left_frame).all()
            assert (from_numpy(right_data[idx % len(left_data)]) == right_frame).all()


def test_crossed_pairs_produces_frames_multiple():
    data = randint(iinfo(int32).max, size=(40, 10, 10))
    left_data, right_data = split(data, (18,))
    left_data1, left_data2 = split(left_data, (8,))
    right_data1, right_data2 = split(right_data, (12,))
    with TemporaryDirectory() as tmpdir:
        left_file_path1 = Path(tmpdir).joinpath("left_testfile1.h5")
        with File(left_file_path1, "w") as left_file1:
            left_file1["left_dataset"] = left_data1
        left_file_path2 = Path(tmpdir).joinpath("left_testfile2.h5")
        with File(left_file_path2, "w") as left_file2:
            left_file2["left_dataset"] = left_data2
        right_file_path1 = Path(tmpdir).joinpath("right_testfile1.h5")
        with File(right_file_path1, "w") as right_file1:
            right_file1["right_dataset"] = right_data1
        right_file_path2 = Path(tmpdir).joinpath("right_testfile2.h5")
        with File(right_file_path2, "w") as right_file2:
            right_file2["right_dataset"] = right_data2
        for idx, (left_frame, right_frame) in enumerate(
            CrossedFramePairs(
                [left_file_path1, left_file_path2],
                [right_file_path1, right_file_path2],
                "left_dataset",
                "right_dataset",
            )
        ):
            assert (from_numpy(left_data[idx // len(left_data)]) == left_frame).all()
            assert (from_numpy(right_data[idx % len(left_data)]) == right_frame).all()


def test_crossed_pairs_reports_length():
    data = randint(iinfo(int32).max, size=(40, 10, 10))
    left_data, right_data = split(data, 2)
    left_data1, left_data2 = split(left_data, (8,))
    with TemporaryDirectory() as tmpdir:
        left_file_path1 = Path(tmpdir).joinpath("left_testfile1.h5")
        with File(left_file_path1, "w") as left_file1:
            left_file1["left_dataset"] = left_data1
        left_file_path2 = Path(tmpdir).joinpath("left_testfile2.h5")
        with File(left_file_path2, "w") as left_file2:
            left_file2["left_dataset"] = left_data2
        right_file_path = Path(tmpdir).joinpath("right_testfile.h5")
        with File(right_file_path, "w") as right_file:
            right_file["right_dataset"] = right_data
        assert 400 == len(
            CrossedFramePairs(
                [left_file_path1, left_file_path2],
                {right_file_path},
                "left_dataset",
                "right_dataset",
            )
        )

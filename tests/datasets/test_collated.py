from pathlib import Path
from tempfile import TemporaryDirectory

from h5py import File
from numpy import iinfo, int32, split
from numpy.random import randint
from pytest import raises
from torch import from_numpy

from ad_denoise.datasets.collated import CrossedDatasets, ZippedDatasets
from ad_denoise.datasets.hdf5 import SimpleHdf5


def test_zipped_produces_frames_single():
    data = randint(iinfo(int32).max, size=(10, 10, 10))
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir).joinpath("testfile.h5")
        with File(file_path, "w") as file:
            file["dataset"] = data
        for idx, ((frame,),) in enumerate(
            ZippedDatasets(SimpleHdf5({file_path}, ["dataset"], [2]))
        ):
            assert (from_numpy(data[idx]) == frame).all()


def test_zipped_produces_frames_multiple():
    data = randint(iinfo(int32).max, size=(30, 10, 10))
    data1, data2, data3 = split(data, 3)
    with TemporaryDirectory() as tmpdir:
        file_path1 = Path(tmpdir).joinpath("testfile1.h5")
        with File(file_path1, "w") as left_file1:
            left_file1["dataset1"] = data1
        file_path2 = Path(tmpdir).joinpath("testfile2.h5")
        with File(file_path2, "w") as left_file2:
            left_file2["dataset2"] = data2
        file_path3 = Path(tmpdir).joinpath("testfile3.h5")
        with File(file_path3, "w") as file3:
            file3["dataset3"] = data3
        for idx, ((frame1,), (frame2,), (frame3,)) in enumerate(
            ZippedDatasets(
                (SimpleHdf5({file_path1}, ["dataset1"], [2])),
                (SimpleHdf5({file_path2}, ["dataset2"], [2])),
                (SimpleHdf5({file_path3}, ["dataset3"], [2])),
            )
        ):
            assert (from_numpy(data1[idx]) == frame1).all()
            assert (from_numpy(data2[idx]) == frame2).all()
            assert (from_numpy(data3[idx]) == frame3).all()


def test_zipped_reports_length():
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
            ZippedDatasets(
                SimpleHdf5({left_file_path1, left_file_path2}, ["left_dataset"], [2]),
                SimpleHdf5({right_file_path}, ["right_dataset"], [2]),
            )
        )


def test_different_lengths_raises_on_init():
    data = randint(iinfo(int32).max, size=(20, 10, 10))
    short_data, long_data = split(data, (8,))
    with TemporaryDirectory() as tmpdir:
        short_file_path = Path(tmpdir).joinpath("short_testfile.h5")
        with File(short_file_path, "w") as short_file:
            short_file["short_dataset"] = short_data
        long_file_path = Path(tmpdir).joinpath("long_testfile.h5")
        with File(long_file_path, "w") as long_file:
            long_file["long_dataset"] = long_data
        with raises(ValueError):
            ZippedDatasets(
                SimpleHdf5({short_file_path}, ["short_dataset"], [2]),
                SimpleHdf5({long_file_path}, ["long_dataset"], [2]),
            )


def test_crossed_produces_frames_single():
    data = randint(iinfo(int32).max, size=(10, 1, 1))
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir).joinpath("testfile.h5")
        with File(file_path, "w") as file:
            file["dataset"] = data
        for idx, ((frame,),) in enumerate(
            CrossedDatasets(SimpleHdf5({file_path}, ["dataset"], [2]))
        ):
            assert (from_numpy(data[idx]) == frame).all()


def test_crossed_produces_frames_multiple():
    data = randint(iinfo(int32).max, size=(6, 10, 10))
    data1, data2, data3 = split(data, (1, 3))
    with TemporaryDirectory() as tmpdir:
        file_path1 = Path(tmpdir).joinpath("testfile1.h5")
        with File(file_path1, "w") as file1:
            file1["dataset1"] = data1
        file_path2 = Path(tmpdir).joinpath("testfile2.h5")
        with File(file_path2, "w") as file2:
            file2["dataset2"] = data2
        file_path3 = Path(tmpdir).joinpath("testfile3.h5")
        with File(file_path3, "w") as file3:
            file3["dataset3"] = data3
        for idx, ((frame1,), (frame2,), (frame3,)) in enumerate(
            CrossedDatasets(
                SimpleHdf5({file_path1}, ["dataset1"], [2]),
                SimpleHdf5({file_path2}, ["dataset2"], [2]),
                SimpleHdf5({file_path3}, ["dataset3"], [2]),
            )
        ):
            assert (from_numpy(data1[0]) == frame1).all()
            assert (from_numpy(data2[idx % 2]) == frame2).all()
            assert (from_numpy(data3[(idx // 2) % 3]) == frame3).all()


def test_crossed_reports_length():
    data = randint(iinfo(int32).max, size=(10, 10, 10))
    data1, data2, data3 = split(data, (2, 5))
    with TemporaryDirectory() as tmpdir:
        file_path1 = Path(tmpdir).joinpath("left_testfile1.h5")
        with File(file_path1, "w") as file1:
            file1["dataset1"] = data1
        file_path2 = Path(tmpdir).joinpath("left_testfile2.h5")
        with File(file_path2, "w") as file2:
            file2["dataset2"] = data2
        file_path3 = Path(tmpdir).joinpath("right_testfile.h5")
        with File(file_path3, "w") as file3:
            file3["dataset3"] = data3
        assert 30 == len(
            CrossedDatasets(
                SimpleHdf5({file_path1}, ["dataset1"], [2]),
                SimpleHdf5({file_path2}, ["dataset2"], [2]),
                SimpleHdf5({file_path3}, ["dataset3"], [2]),
            )
        )

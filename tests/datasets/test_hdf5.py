from pathlib import Path
from tempfile import TemporaryDirectory

from h5py import File
from numpy import atleast_1d, iinfo, int32, split
from numpy.random import randint
from torch import from_numpy

from ad_denoise.datasets.hdf5 import SimpleHdf5


def test_simple_hdf5_produces_frames():
    data = randint(iinfo(int32).max, size=(10, 10, 10))
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir).joinpath("testfile.h5")
        with File(file_path, "w") as file:
            file["dataset"] = data
        for idx, (frame,) in enumerate(SimpleHdf5([file_path], ["dataset"], [2])):
            assert (from_numpy(data[idx]) == frame).all()


def test_simple_hdf5_produces_frames_multiple_paths():
    data = randint(iinfo(int32).max, size=(20, 10, 10))
    data1, data2 = split(data, (8,))
    with TemporaryDirectory() as tmpdir:
        file_path1 = Path(tmpdir).joinpath("testfile1.h5")
        with File(file_path1, "w") as file1:
            file1["dataset"] = data1
        file_path2 = Path(tmpdir).joinpath("testfile2.h5")
        with File(file_path2, "w") as file2:
            file2["dataset"] = data2
        for idx, (frame,) in enumerate(
            SimpleHdf5([file_path1, file_path2], ["dataset"], [2])
        ):
            assert (from_numpy(data[idx]) == frame).all()


def test_simple_hdf5_produces_frames_multiple_keys():
    data1 = randint(iinfo(int32).max, size=(20, 10, 10))
    data2 = randint(iinfo(int32).max, size=(20,))
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir).joinpath("testfile.h5")
        with File(file_path, "w") as file:
            file["dataset1"] = data1
            file["dataset2"] = data2
        for idx, (frame1, frame2) in enumerate(
            SimpleHdf5([file_path], ["dataset1", "dataset2"], [2, 0])
        ):
            assert (from_numpy(data1[idx]) == frame1).all()
            assert (from_numpy(atleast_1d(data2[idx])) == frame2).all()


def test_simple_hdf5_reports_length():
    data = randint(iinfo(int32).max, size=(20, 10, 10))
    data1, data2 = split(data, (8,))
    with TemporaryDirectory() as tmpdir:
        file_path1 = Path(tmpdir).joinpath("testfile1.h5")
        with File(file_path1, "w") as file1:
            file1["dataset"] = data1
        file_path2 = Path(tmpdir).joinpath("testfile2.h5")
        with File(file_path2, "w") as file2:
            file2["dataset"] = data2
        assert 20 == len(SimpleHdf5([file_path1, file_path2], ["dataset"], [2]))

from pathlib import Path
from tempfile import TemporaryDirectory

from h5py import File
from numpy import iinfo, int32, split
from numpy.random import randint
from torch import from_numpy

from ad_denoise.datasets.single_frames import SingleFrames


def test_single_frame_produces_frames():
    data = randint(iinfo(int32).max, size=(10, 10, 10))
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir).joinpath("testfile.h5")
        with File(file_path, "w") as file:
            file["dataset"] = data
        for idx, frame in enumerate(SingleFrames({file_path}, "dataset")):
            assert (from_numpy(data[idx]) == frame).all()


def test_single_frame_produces_frames_multiple():
    data = randint(iinfo(int32).max, size=(20, 10, 10))
    data1, data2 = split(data, (8,))
    with TemporaryDirectory() as tmpdir:
        file_path1 = Path(tmpdir).joinpath("testfile1.h5")
        with File(file_path1, "w") as file1:
            file1["dataset"] = data1
        file_path2 = Path(tmpdir).joinpath("testfile2.h5")
        with File(file_path2, "w") as file2:
            file2["dataset"] = data2
        for idx, frame in enumerate(SingleFrames([file_path1, file_path2], "dataset")):
            assert (from_numpy(data[idx]) == frame).all()


def test_single_frame_reports_length():
    data = randint(iinfo(int32).max, size=(20, 10, 10))
    data1, data2 = split(data, (8,))
    with TemporaryDirectory() as tmpdir:
        file_path1 = Path(tmpdir).joinpath("testfile1.h5")
        with File(file_path1, "w") as file1:
            file1["dataset"] = data1
        file_path2 = Path(tmpdir).joinpath("testfile2.h5")
        with File(file_path2, "w") as file2:
            file2["dataset"] = data2
        assert 20 == len(SingleFrames([file_path1, file_path2], "dataset"))

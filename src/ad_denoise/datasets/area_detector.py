from typing import cast

from torch import Tensor

from .collated import ZippedDatasets
from .computed import ComputedFramesDataset
from .hdf5 import H5Key, H5Path, H5Paths, SimpleHdf5
from .repeating import RepeatingDataset
from .utils import Dim, SizedDataset


class Hdf5AreaDetectorImages(
    ComputedFramesDataset[tuple[Tensor, Tensor, Tensor], Tensor]
):
    """A high level pytorch dataset for loading area detecor images from hdf5."""

    def __init__(
        self,
        data_paths: H5Paths,
        frame_key: H5Key,
        count_times_key: H5Key,
        mask_path: H5Path,
        mask_key: H5Key,
    ) -> None:
        """Creates a high level dataset of masked, normalized frames from hdf5.

        Args:
            data_paths: A sequence of paths to hdf5 files containing frames and count
                times.
            frame_key: The key which locates the detector data within the hdf5 files.
            count_times_key: The key which locates the detector data within the hdf5
                files.
            mask_path (H5Path): The path to a file containing the frame mask.
            mask_key (H5Key): The key which locates the frame mask within the hdf5 file.
        """
        frames_dataset = SimpleHdf5(data_paths, frame_key, Dim(2))
        frame_times_dataset = SimpleHdf5(data_paths, count_times_key, Dim(0))
        mask_dataset = RepeatingDataset(
            SimpleHdf5((mask_path,), mask_key, Dim(2)), len(frames_dataset)
        )
        super(Hdf5AreaDetectorImages, self).__init__(
            cast(
                SizedDataset[tuple[Tensor, Tensor, Tensor]],
                ZippedDatasets(
                    frames_dataset,
                    frame_times_dataset,
                    mask_dataset,
                ),
            ),
            Hdf5AreaDetectorImages._mask_and_normalize,
        )

    @staticmethod
    def _mask_and_normalize(frame: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        return frame[0] * frame[1] / frame[2]

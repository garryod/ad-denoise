ad_denoise
==========

|code_ci| |docs_ci| |coverage| |license|

This package provides the tools nessacary to train a neural network to denoise
syncrotron x-ray area detector images.

============== ==============================================
Source code    https://github.com/garryod/ad_denoise
Documentation  https://garryod.github.io/ad_denoise
Releases       https://github.com/garryod/ad_denoise/releases
============== ==============================================

Model training can be performed using the below command:

.. code:: bash

    python -m ad_denoise train my_config.yaml

Where ``my_config.yaml`` is as so:

.. code:: yaml

    max_epochs: 50
    model:
        Noise2Self:
            network:
                Gaussian:
                    kernel_half_width: 3
                    train_dataset:
                        Hdf5ADImagesDataset:
                            data_paths:
                            - /dls/i22/data/2022/cm31149-3/Denoising/i22-629817.nxs
                            frame_key: entry1/detector/data
                            count_times_key: entry1/instrument/detector/count_time
                            mask_path: /dls/i22/data/2022/cm31149-3/processing/SAXS_mask.nxs
                            mask_key: entry/mask/mask
                    val_dataset:
                        InputTargetDataset:
                            input:
                                Hdf5ADImagesDataset:
                                    data_paths:
                                    - /dls/i22/data/2022/cm31149-3/Denoising/i22-629817.nxs
                                    frame_key: entry1/detector/data
                                    count_times_key: entry1/instrument/detector/count_time
                                    mask_path: /dls/i22/data/2022/cm31149-3/processing/SAXS_mask.nxs
                                    mask_key: entry/mask/mask
                            target:
                                Hdf5ADImagesDataset:
                                    data_paths:
                                    - /dls/i22/data/2022/cm31149-3/Denoising/i22-629822.nxs
                                    - /dls/i22/data/2022/cm31149-3/Denoising/i22-629823.nxs
                                    frame_key: entry1/detector/data
                                    count_times_key: entry1/instrument/detector/count_time
                                    mask_path: /dls/i22/data/2022/cm31149-3/processing/SAXS_mask.nxs
                                    mask_key: entry/mask/mask

.. |code_ci| image:: https://github.com/garryod/ad_denoise/workflows/Code%20CI/badge.svg?branch=main
    :target: https://github.com/garryod/ad_denoise/actions?query=workflow%3A%22Code+CI%22
    :alt: Code CI

.. |docs_ci| image:: https://github.com/garryod/ad_denoise/workflows/Docs%20CI/badge.svg?branch=main
    :target: https://github.com/garryod/ad_denoise/actions?query=workflow%3A%22Docs+CI%22
    :alt: Docs CI

.. |coverage| image:: https://codecov.io/gh/garryod/ad_denoise/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/garryod/ad_denoise
    :alt: Test Coverage

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
    :alt: Apache License

..
    Anything below this line is used when viewing README.rst and will be replaced
    when included in index.rst

See https://garryod.github.io/ad_denoise for more detailed documentation.

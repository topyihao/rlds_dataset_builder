from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
sys.path.append('.')
from aloha1_put_X_into_pot_300_demos.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(episode_path):
        # Load raw data
        with h5py.File(episode_path, "r") as F:
            actions = F["/action"][()]
            states = F["/observations/qpos"][()]
            images = F["/observations/images/cam_high"][()]  # Primary camera (top-down view)
            right_wrist_images = F["/observations/images/cam_right_wrist"][()]  # Right wrist camera

        # Get language instruction from filename
        # Assumes filepaths look like: "/PATH/TO/ALOHA/PREPROCESSED/DATASETS/put_sponge_into_pot/train/episode_0.hdf5"
        raw_file_string = episode_path.split('/')[-3]  # Extract "put_sponge_into_pot"
        command = " ".join(raw_file_string.split("_"))  # Convert to "put sponge into pot"

        # Assemble episode: here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'image': images[i],
                    'wrist_image': right_wrist_images[i],
                    'state': np.asarray(states[i], np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
            })

        # Create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # If you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # For smallish datasets, use single-thread parsing
    for sample in paths:
        ret = _parse_example(sample)
        yield ret


class AlohaPutSpongeIntoPot(MultiThreadedDatasetBuilder):
    """DatasetBuilder for ALOHA put sponge into pot dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 10             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 20   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on available RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation (cam_high).',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint state (7D arm qpos).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot arm action (7D).',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob("/home/allied/aloha_data/aloha_preprocessed/put_sponge_into_pot/train/*.hdf5"),
            "val": glob.glob("/home/allied/aloha_data/aloha_preprocessed/put_sponge_into_pot/val/*.hdf5"),
        } 
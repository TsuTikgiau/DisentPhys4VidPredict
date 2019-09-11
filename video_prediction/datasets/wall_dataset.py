import itertools
import os
import re

import tensorflow as tf

from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset


class WallDataset(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(WallDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'images/encoded', (128, 128, 3)
        self.state_like_names_and_shapes['speed'] = 'speed', [1]
        self.state_like_names_and_shapes['position'] = 'position', [1]
        self.static_state['setting'] = 'setting', [5]

    def get_default_hparams_dict(self):
        default_hparams = super(WallDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=4,
            sequence_length=10,
            random_crop_size=0,
            use_state=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return True

    def decode_and_preprocess_images(self, image_buffers, image_shape):
        if self.hparams.crop_size:
            raise NotImplementedError
        if self.hparams.scale_size:
            raise NotImplementedError
        image_buffers = tf.reshape(image_buffers, [-1])
        if not isinstance(image_buffers, (list, tuple)):
            image_buffers = tf.unstack(image_buffers)
        image_size = tf.image.extract_jpeg_shape(image_buffers[0])[:2]  # should be the same as image_shape[:2]
        if self.hparams.random_crop_size:
            random_crop_size = [self.hparams.random_crop_size] * 2
            crop_y = tf.random_uniform([], minval=0, maxval=image_size[0] - random_crop_size[0], dtype=tf.int32)
            crop_x = tf.random_uniform([], minval=0, maxval=image_size[1] - random_crop_size[1], dtype=tf.int32)
            crop_window = [crop_y, crop_x] + random_crop_size
            images = [tf.image.decode_and_crop_jpeg(image_buffer, crop_window) for image_buffer in image_buffers]
            images = tf.image.convert_image_dtype(images, dtype=tf.float32)
            images.set_shape([None] + random_crop_size + [image_shape[-1]])
        else:
            images = [tf.image.decode_jpeg(image_buffer) for image_buffer in image_buffers]
            images = tf.image.convert_image_dtype(images, dtype=tf.float32)
            images.set_shape([None] + list(image_shape))
        # TODO: only random crop for training
        return images

    def num_examples_per_epoch(self):
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        for filename in self.filenames:
            match = re.search('sequence_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count


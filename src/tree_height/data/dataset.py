import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
KERNEL_BUFFER = 128

opticalBands = ['B2', 'B3', 'B4', 'B8']
redEdgeBands = ['B5', 'B6', 'B7', 'B8A']
BANDS = opticalBands + redEdgeBands

RESPONSE = 'rh98'
FEATURES = BANDS + [RESPONSE]


class TFRecordDataset(Dataset):
    x_mean = (0.031897928565740585, 0.052805472165346146, 0.046148691326379776, 0.2097931057214737,
              0.09050391614437103, 0.1770937293767929, 0.1983453631401062, 0.21818380057811737)
    x_std = (0.013345479033887386, 0.012756776995956898, 0.015102902427315712, 0.04132801294326782,
             0.014108794741332531, 0.035968560725450516, 0.041470590978860855, 0.04269599914550781)

    y_mean = (13.005148887634277,)
    y_std = (5.81528902053833,)
    y_nodata = 0

    image_transform = T.Normalize(mean=x_mean, std=x_std)
    gt_transform = T.Normalize(mean=y_mean, std=y_std)
    gt_transform_invert = T.Compose([T.Normalize(mean=(0,),
                                                 std=(1/i for i in y_std)),
                                     T.Normalize(mean=(-i for i in y_mean),
                                                 std=(1,)),
                                     ])

    def __init__(self, tfrecord_files: list[str], is_predict: bool,):

        self.tfrecord_files = tfrecord_files
        self.is_predict = is_predict
        self.dataset = self._load_tfrecord()

    def _load_tfrecord(self):
        dataset = []
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_files, compression_type='GZIP')

        if self.is_predict:
            PAD_SHAPE = [a + b for a, b in zip(KERNEL_SHAPE, [KERNEL_BUFFER, KERNEL_BUFFER])]
            COLUMNS = [
                tf.io.FixedLenFeature(shape=PAD_SHAPE, dtype=tf.float32) for k in BANDS
            ]
            BANDS_DICT = dict(zip(BANDS, COLUMNS))

            for raw_record in raw_dataset:
                parsed_bands = tf.io.parse_single_example(raw_record, BANDS_DICT)
                bands = [parsed_bands[band].numpy() for band in BANDS]
                image_data = torch.tensor(np.array(bands), dtype=torch.float32)
                if self.image_transform:
                    image_data = self.image_transform(image_data)
                dataset.append(image_data)

        if not self.is_predict:
            COLUMNS = [
                tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
            ]
            FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

            for raw_record in raw_dataset:
                parsed_features = tf.io.parse_single_example(raw_record, FEATURES_DICT)
                bands = [parsed_features[band].numpy() for band in BANDS]
                label = [parsed_features[RESPONSE].numpy()]

                image_data = torch.tensor(np.array(bands), dtype=torch.float32)
                if self.image_transform:
                    image_data = self.image_transform(image_data)

                gt_data = torch.tensor(np.array(label), dtype=torch.float32)
                gt_data[gt_data == self.y_nodata] = float('NaN')
                if self.image_transform:
                    gt_data = self.gt_transform(gt_data)

                dataset.append((image_data, gt_data))

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == '__main__':
    import glob
    trainTFRecordPath = glob.glob(os.path.join('/mnt/code/deep_learning/tree_height/dataset/train', '*.tfrecord.gz'))
    trainTFRecordDataset = TFRecordDataset(trainTFRecordPath, is_predict=False)

    for x, y in trainTFRecordDataset:
        print(torch.mean(x), torch.std(x))
        y = y[~torch.isnan(y)]
        print(torch.mean(y), torch.std(y))




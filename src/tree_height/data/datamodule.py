from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import os
import glob

from tree_height.data.dataset import TFRecordDataset


class TFRecordDataModule(LightningDataModule):
    def __init__(self,
                 root_path: str,
                 batch_size: int = 20):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

    def setup(self, stage: [str] = None):
        if stage == "fit":

            train_files = glob.glob(os.path.join(self.root_path, "train", '*.tfrecord.gz'))
            self.train_dataset = TFRecordDataset(tfrecord_files=train_files, is_predict=False)

            val_files = glob.glob(os.path.join(self.root_path, "val", '*.tfrecord.gz'))
            self.val_dataset = TFRecordDataset(tfrecord_files=val_files, is_predict=False)

        if stage == "predict":
            predict_files = glob.glob(os.path.join(self.root_path, "predict", '*.tfrecord.gz'))
            self.predict_dataset = TFRecordDataset(tfrecord_files=predict_files, is_predict=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1, shuffle=False)


if __name__ == '__main__':
    module = TFRecordDataModule(root_path='/mnt/code/deep_learning/tree_height/dataset')
    module.setup("fit")
    

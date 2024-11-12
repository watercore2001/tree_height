from pytorch_lightning import LightningModule
import torch
from torch import nn
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import datetime
from tree_height.train.optimizer import OptimizerArgs, CosineAnnealingWithWarmup
from tree_height.data.dataset import KERNEL_SIZE, TFRecordDataset
import os
import tifffile
import rasterio
import json
import numpy as np
from typing import Any

class TreeHeightModule(LightningModule):

    def __init__(self,
                 model: nn.Module,
                 optim_args: OptimizerArgs,
                 save_path: str,):
        super().__init__()
        self.model = model
        self.optim_args = optim_args

        # loss
        self.loss = nn.L1Loss()

        # metrics
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()

        # save predict out
        now = datetime.datetime.now()
        self.save_dir = os.path.join(save_path, now.strftime("%Y-%m-%d_%H-%M"))
        self.mosaic_json = os.path.join(self.save_dir, 'FCNN_demo_genhe_-mixer.json')
        self.save_x_path = os.path.join(self.save_dir, 'x')
        self.save_y_path = os.path.join(self.save_dir, 'y')
        os.makedirs(self.save_x_path, exist_ok=True)
        os.makedirs(self.save_y_path, exist_ok=True)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: dict, batch_index: int):
        x, y = batch
        gt_mask = ~torch.isnan(y)
        y = y[gt_mask]

        y_hat = self(x)
        y_hat = y_hat[gt_mask]
        loss = self.loss(y_hat, y)

        self.log(name="train_loss", value=loss, on_step=True)

        return loss

    def validation_step(self, batch: dict, batch_index: int):
        x, y = batch
        gt_mask = ~torch.isnan(y)
        y = y[gt_mask]

        y_hat = self(x)
        y_hat = y_hat[gt_mask]
        loss = self.loss(y_hat, y)

        self.log(name="val_loss", value=loss, on_epoch=True)

        self.mse(y_hat, y)
        self.log('val_mse', self.mse, on_step=True, on_epoch=True)

        self.mae(y_hat, y)
        self.log('val_mae', self.mae, on_step=True, on_epoch=True)

        self.r2(y_hat, y)
        self.log('val_r2', self.r2, on_step=True, on_epoch=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):

        def save_image(image: torch.Tensor, save_path: str):
            np_image = image.numpy()[0]
            _, h, w = np_image.shape
            top = (h - KERNEL_SIZE) // 2
            left = (w - KERNEL_SIZE) // 2

            result = np_image[0:3, top:top + KERNEL_SIZE, left:left + KERNEL_SIZE]
            tifffile.imwrite(save_path, result)

            return 0

        x = batch
        y_hat = self(x)
        un_normalized_y = TFRecordDataset.gt_transform_invert(y_hat)

        x_path = os.path.join(self.save_x_path, f"{batch_idx}.tif")
        y_path = os.path.join(self.save_y_path, f"{batch_idx}.tif")

        save_image(x, save_path=x_path)
        save_image(un_normalized_y, save_path=y_path)

    def on_predict_epoch_end(self):

        with open(self.mosaic_json) as file:
            jsonData = json.load(file)

        patchWidth, patchHeight = jsonData['patchDimensions']

        totalPatches = jsonData['totalPatches']
        patchRow = jsonData['patchesPerRow']
        patchCol = totalPatches // patchRow

        affineMat = jsonData['projection']['affine']['doubleMatrix']
        transform = rasterio.Affine(*affineMat)

        imgWidth = patchWidth * patchRow
        imgHeight = patchHeight * patchCol

        fullImg = np.zeros((imgHeight, imgWidth), dtype=np.float32)

        for i in range(totalPatches):
            patch_i_path = os.path.join(self.save_y_path, f"{i}.tif")  # each patch

            # 计算每个 patch 在完整图像中的位置
            patch_i_row = i // patchRow
            patch_i_col = i % patchRow
            xOffset = patch_i_col * patchWidth
            yOffset = patch_i_row * patchHeight

            with rasterio.open(patch_i_path) as src:
                patch_i = src.read(1)

            fullImg[yOffset:yOffset + patchHeight, xOffset:xOffset + patchWidth] = patch_i

        # 保存结果
        mosaicImgDir = os.path.join(self.save_dir, 'mosaic.tif')
        with rasterio.open(mosaicImgDir, 'w', driver='GTiff',
                           height=fullImg.shape[0], width=fullImg.shape[1],
                           count=1, dtype=fullImg.dtype,
                           crs=jsonData['projection']['crs'],
                           transform=transform) as dst:
            dst.write(fullImg, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.optim_args.max_lr,
                                      weight_decay=self.optim_args.weight_decay)

        lr_scheduler = CosineAnnealingWithWarmup(optimizer=optimizer,
                                                 warmup_epochs=self.optim_args.warmup_epochs,
                                                 annealing_epochs=self.optim_args.annealing_epochs,
                                                 max_lr=self.optim_args.max_lr,
                                                 min_lr=self.optim_args.min_lr)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

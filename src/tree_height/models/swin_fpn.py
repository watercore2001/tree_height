import timm
import torch
from einops import rearrange
from tree_height.models.fpn import FpnDecoder
from torch import nn
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class SwinFPNTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 4
        self.swin = timm.create_model(model_name='swinv2_tiny_window8_256.ms_in1k',
                                      checkpoint_path="",
                                      in_chans=8, features_only=True, patch_size=self.patch_size,
                                      drop_rate=0.2, drop_path_rate=0.2)
        print(self.swin.default_cfg)
        self.fpn = FpnDecoder(input_dims=[96, 192, 384, 768], inner_dim=96, seg_dim=96, output_dim=96)
        self.header = nn.Linear(in_features=96, out_features=self.patch_size**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features: [(b, h1, w1, c1), (b, h2, w2, c2), ..]
        features = self.swin(x)
        for i in range(len(features)):
            features[i] = rearrange(features[i], "b h w c -> b c h w")

        x = self.fpn(features)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.header(x)
        x = rearrange(x, "b h w (p1 p2) -> b 1 (h p1) (w p2)", p1=self.patch_size, p2=self.patch_size)
        return x


if __name__ == '__main__':

    # model = timm.create_model(model_name='swinv2_tiny_window8_256.ms_in1k', pretrained=False,
    #                           in_chans=8, features_only=True, patch_size=4,
    #                           drop_rate=0.2, drop_path_rate=0.2)
    model = SwinFPNTiny()
    input = torch.rand(1, 8, 256, 256)
    output = model(input)
    print(output.shape)
    # for o in output:
    #     print(o.shape)

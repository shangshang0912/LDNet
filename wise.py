import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepWise_PointWise_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(DeepWise_PointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
if __name__ == '__main__':
    input = torch.randn(4, 3, 256, 256)
    block = DeepWise_PointWise_Conv(3, 3, 3)
    out = block(input)
    print(out.size())
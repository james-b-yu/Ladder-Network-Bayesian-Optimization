from typing import NamedTuple
import torch as t
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.utils.data

DEFAULT_STD = 0.1


class AutoEncoderFModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.corruption_std = DEFAULT_STD  # to be overwritten when calling module.apply on the parent module


class AutoEncoderFConvBlock(AutoEncoderFModule):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dummy_param = nn.Parameter(t.empty(0))  # for finding the device

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)
        self.nonlinearity = nn.ReLU()

    def forward(self, x, corrupt: bool = False):
        out = self.batch_norm(x)
        out = x
        out = self.conv(out)
        if corrupt and self.training:
            out = out + t.normal(0, self.corruption_std, out.shape, device=self.dummy_param.device)

        out = self.nonlinearity(out)
        return out


class AutoEncoderFlatternBlock(AutoEncoderFModule):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(t.empty(0))  # for finding the device

        self.flatten = nn.Flatten()

    def forward(self, x, corrupt: bool = False):
        out = self.flatten(x)
        if corrupt and self.training:
            out = out + t.normal(0, self.corruption_std, out.shape, device=self.dummy_param.device)

        return out


class AutoEncoderDeepFLayer(AutoEncoderFModule):
    def __init__(self, in_features: int, out_features: int, nonlinearity: bool):
        super().__init__()
        self.dummy_param = nn.Parameter(t.empty(0))  # for finding the device

        self.in_features = in_features
        self.out_features = out_features

        self.layer = nn.Sequential(
            # nn.BatchNorm1d(num_features=in_features),  # batch_norm
            nn.Linear(in_features, out_features)
        )

        self.nonlinearity = nn.Tanh() if nonlinearity else nn.Identity()

    def forward(self, x, corrupt: bool = False):
        out = self.layer(x)
        if corrupt and self.training:
            out = out + t.normal(0, self.corruption_std, out.shape, device=self.dummy_param.device)
        else:
            out = self.nonlinearity(out)
        return out


class AutoEncoderDeepBLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.dummy_param = nn.Parameter(t.empty(0))  # for finding the device

        self.in_features = in_features
        self.out_features = out_features

        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(num_features=out_features)  # batch_norm
        )

        self.biases = nn.Parameter(t.zeros(2, out_features, requires_grad=True))
        self.xweights = nn.Parameter(t.ones(2, out_features, requires_grad=True))
        self.uweights = nn.Parameter(t.zeros(2, out_features, requires_grad=True))
        self.crossweights = nn.Parameter(t.zeros(2, out_features, requires_grad=True))
        self.sweights = nn.Parameter(t.ones(1, out_features, requires_grad=True))

    def forward(self, x, cl):  # x: z^{l + 1}, cl: corrupted z
        u = self.layer(x)

        out = self.biases[0] + self.xweights[0] * cl + self.uweights[0] * u + self.crossweights[0] * cl * u + self.sweights * t.sigmoid(self.biases[1] + self.xweights[1] * cl + self.uweights[1] * u + self.crossweights[1] * cl * u)

        return out


class AutoEncoderBConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, in_features: tuple[int, int], out_features: tuple[int, int]):
        super().__init__()
        self.dummy_param = nn.Parameter(t.empty(0))  # for finding the device

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_features = in_features
        self.out_features = out_features

        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, output_padding=1 if 2 * in_features[0] + 1 != out_features[0] else 0)

        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

        self.biases = nn.Parameter(t.zeros(2, *out_features, requires_grad=True))
        self.xweights = nn.Parameter(t.ones(2, *out_features, requires_grad=True))
        self.uweights = nn.Parameter(t.zeros(2, *out_features, requires_grad=True))
        self.crossweights = nn.Parameter(t.zeros(2, *out_features, requires_grad=True))
        self.sweights = nn.Parameter(t.ones(1, *out_features, requires_grad=True))

    def forward(self, x, cl):
        u = self.conv(x)
        u = self.batch_norm(u)
        out = self.biases[0] + self.xweights[0] * cl + self.uweights[0] * u + self.crossweights[0] * cl * u + self.sweights * t.sigmoid(self.biases[1] + self.xweights[1] * cl + self.uweights[1] * u + self.crossweights[1] * cl * u)
        return out


class AutoEncoderModel(nn.Module):
    def __init__(self, input_shape: tuple[int, int], output_classes: int):
        super().__init__()
        self.corruption_std = DEFAULT_STD

        self.dummy_param = nn.Parameter(t.empty(0))  # for finding the device

        self.tail = nn.BatchNorm2d(num_features=1)  # batch_norm

        conv_backbone_encoder_arr = [AutoEncoderFConvBlock(in_channels=1, out_channels=4),
                                     AutoEncoderFConvBlock(in_channels=4, out_channels=8),
                                     AutoEncoderFConvBlock(in_channels=8, out_channels=16)]

        self.conv_backbone_encoder = nn.ModuleList(conv_backbone_encoder_arr)
        test_input = t.rand(input_shape).unsqueeze(0).unsqueeze(0)
        test_conv_backbone = nn.Sequential(*conv_backbone_encoder_arr)
        self.conv_backbone_encoder_output_shape: t.Size = t.Size((-1, *test_conv_backbone(self.tail(test_input)).shape[1:]))
        self.conv_backbone_encoder_output_size: int = test_conv_backbone(self.tail(test_input)).flatten(start_dim=1).size(1)

        deep_backbone_encoder_arr = [AutoEncoderDeepFLayer(self.conv_backbone_encoder_output_size, 128, True), AutoEncoderDeepFLayer(128, output_classes, False)]

        self.deep_backbone_encoder = nn.ModuleList(deep_backbone_encoder_arr)

        # we're not going to use sequential since we need more control
        # we also need to find out how many features the thing has
        test_input_mutable = test_input.clone()
        num_features = [test_input_mutable.shape[-2:]]
        for i in conv_backbone_encoder_arr:
            test_input_mutable = i(test_input_mutable)
            num_features.append(test_input_mutable.shape[-2:])

        num_features.reverse()

        self.conv_backbone_decoder = nn.ModuleList([AutoEncoderBConvBlock(conv.out_channels, conv.in_channels, num_features[index], num_features[index + 1]) for index, conv in enumerate(reversed(conv_backbone_encoder_arr))])
        self.deep_backbone_decoder = nn.ModuleList([AutoEncoderDeepBLayer(deep.out_features, deep.in_features) for deep in reversed(deep_backbone_encoder_arr)])

    def forward(self, x) -> tuple[list[t.Tensor], list[t.Tensor]]:

        # *** carry out encoder corrupted calculation ***
        # *** conv layers ***
        corrupted_conv_backbone_outputs = [self.tail(x) + t.normal(0, self.corruption_std, x.shape, device=self.dummy_param.device) if self.training else t.zeros(x.shape)]  # includes input layer

        for m in self.conv_backbone_encoder:
            corrupted_conv_backbone_outputs.append(m(corrupted_conv_backbone_outputs[-1], True))

        flattened_corrupted_conv_backbone_output = corrupted_conv_backbone_outputs[-1].flatten(start_dim=1)  # flattened output of final conv layer
        # *** deep layers ***
        deep_layers = self.deep_backbone_encoder.children()
        first_deep_layer = next(deep_layers)

        corrupted_deep_backbone_outputs = [first_deep_layer(flattened_corrupted_conv_backbone_output, True)]  # corrupted outputs without the nonlinearity
        for m in deep_layers:
            corrupted_deep_backbone_outputs.append(m(corrupted_deep_backbone_outputs[-1], True))

        # *** carry out deep layer decoder calculation ***
        ladder_down = [corrupted_deep_backbone_outputs[-1]]
        for index, m in enumerate(reversed(corrupted_deep_backbone_outputs[1:])):  # iterate down from output layer to first deep layer, but not including first deep layer
            cl_side = corrupted_deep_backbone_outputs[- index - 2]
            ladder_down.append(self.deep_backbone_decoder[index](ladder_down[-1], cl_side))

        decoded_conv_backbone_output = self.deep_backbone_decoder[-1](ladder_down[-1], flattened_corrupted_conv_backbone_output).reshape(self.conv_backbone_encoder_output_shape)  # attempt to denoise corrupted version of conv backbone final layer

        # *** carry out conv layer decoder calculation ***
        ladder_down.append(decoded_conv_backbone_output)
        for index, m in enumerate(self.conv_backbone_decoder):
            cl_side = corrupted_conv_backbone_outputs[- index - 2]
            ladder_down.append(m(ladder_down[-1], cl_side))

        ladder_down.reverse()

        ladder_up = [self.tail(x)]
        for m in self.conv_backbone_encoder:
            ladder_up.append(m(ladder_up[-1]))

        # need to save the index where we convert between flattened and not flattened
        bridge_index = len(ladder_up) - 1
        ladder_up[bridge_index] = ladder_up[bridge_index].flatten(start_dim=1)

        for m in self.deep_backbone_encoder:
            ladder_up.append(m(ladder_up[-1]))

        # convert flattened index back into convolution layout
        ladder_up[bridge_index] = ladder_up[bridge_index].reshape(self.conv_backbone_encoder_output_shape)

        return ladder_up, ladder_down

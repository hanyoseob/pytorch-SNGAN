import torch
import torch.nn as nn
from torch.nn import Parameter

from torch.nn.utils import *


class CNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=[], bias=[], snorm=False):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, snorm=snorm)]

        # if snorm:
        #     layers += [SpectralNorm(layers[-1].conv)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class DECNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, norm='bnorm', relu=0.0, drop=[], bias=[], snorm=False):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Deconv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias, snorm=snorm)]

        # if snorm:
        #     layers += [SpectralNorm(layers[-1].deconv)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.decbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.decbr(x)


class ResBlock(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm='inorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []

        # 1st conv
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        # 2nd conv
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=[])]

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)


class CNR1d(nn.Module):
    def __init__(self, nch_in, nch_out, norm='bnorm', relu=0.0, drop=[]):
        super().__init__()

        if norm == 'bnorm':
            bias = False
        else:
            bias = True

        layers = []
        layers += [nn.Linear(nch_in, nch_out, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True, snorm=False):
        super(Conv2d, self).__init__()
        if snorm:
            # self.conv = SpectralNorm(nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            self.conv = spectral_norm(nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        else:
            self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Deconv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, bias=True, snorm=False):
        super(Deconv2d, self).__init__()
        if snorm:
            # self.deconv = SpectralNorm(nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
            self.deconv = spectral_norm(nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
        else:
            self.deconv = nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

        # self.deconv = nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

        # layers = [nn.Upsample(scale_factor=2, mode='bilinear'),
        #           nn.ReflectionPad2d(1),
        #           nn.Conv2d(nch_in , nch_out, kernel_size=3, stride=1, padding=0)]
        #
        # self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv(x)


class Linear(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(nch_in, nch_out)

    def forward(self, x):
        return self.linear(x)


class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)


class Padding(nn.Module):
    def __init__(self, padding, padding_mode='zeros', value=0):
        super(Padding, self).__init__()
        if padding_mode == 'reflection':
            self. padding = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_mode == 'constant':
            self.padding = nn.ConstantPad2d(padding, value)
        elif padding_mode == 'zeros':
            self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        return self.padding(x)


class Pooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='avg'):
        super().__init__()

        if type == 'avg':
            self.pooling = nn.AvgPool2d(pool)
        elif type == 'max':
            self.pooling = nn.MaxPool2d(pool)
        elif type == 'conv':
            self.pooling = nn.Conv2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pooling(x)


class UnPooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='nearest'):
        super().__init__()

        if type == 'nearest':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='nearest', align_corners=True)
        elif type == 'bilinear':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='bilinear', align_corners=True)
        elif type == 'conv':
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.unpooling(x)


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2,
        #                 diffy // 2, diffy - diffy // 2])

        return torch.cat([x2, x1], dim=1)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X (W*H) X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (W*H)
        energy = torch.bmm(proj_query, proj_key)  # bmm: Batch Matrix Multiplication
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X (W*H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C * (W*H)
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


## LOSS layer

class TV1dLoss(nn.Module):
    def __init__(self):
        super(TV1dLoss, self).__init__()

    def forward(self, input):
        # loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
        #        torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        loss = torch.mean(torch.abs(input[:, :-1] - input[:, 1:]))

        return loss


class TV2dLoss(nn.Module):
    def __init__(self):
        super(TV2dLoss, self).__init__()

    def forward(self, input):
        loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
               torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        return loss


class SSIM2dLoss(nn.Module):
    def __init__(self):
        super(SSIM2dLoss, self).__init__()

    def forward(self, input, targer):
        loss = 0
        return loss


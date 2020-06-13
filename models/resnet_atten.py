import torch
import torch.nn as nn


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnext50_32x4d_attention', 'resnext101_32x8d_attention', 'resnext101_32x16d_wsl_attention',
           'resnext101_32x32d_wsl_attention', 'resnext101_32x48d_wsl_attention']

model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # 基础块

    expansion = 1  # 对输出深度的倍乘

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 归一化
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            # 基础块不能进行空洞卷积
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # 当步长不为1时  self.conv2 和 self.downsample 都会进行downsample

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        # 短路连接信号
        # _make_layer中的downsample
        self.downsample = downsample
        self.stride = stride  # 获取步长

    def forward(self, x):
        identity = x  # 用于短路连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 短路连接
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 残差块

    expansion = 4  # 对输出深度的倍乘

    # 若我们输入深度为64，那么扩张4倍后就变为了256
    # 其目的在于使得当前块的输出深度与下一个块的输入深度保持一致

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 归一化
        width = int(planes * (base_width / 64.)) * groups  # 宽度

        # 当步长不为1时  self.conv2 和 self.downsample 都会进行downsample

        # 这层1*1卷积层，是为了降维，把输出深度降到与3*3卷积层的输入深度一致
        self.conv1 = conv1x1(inplanes, width)
        # 归一化
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # 短路连接信号
        # _make_layer中的downsample
        self.downsample = downsample
        self.stride = stride  # 获取步长

    def forward(self, x):
        identity = x  # 用于短路连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 短路连接
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    # 通道注意力

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2(avg_out)
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2(max_out)
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    # 空间注意力
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # layer层中的输入维度，残差块块输入深度
        self.dilation = 1  # 空洞卷积系数（间隔1）

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups  # 分组卷积数
        self.base_width = width_per_group  # 每组大小

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # self.ca = ChannelAttention(self.inplanes)
        # self.sa = SpatialAttention()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.ca1 = ChannelAttention(self.inplanes)
        self.sa1 = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():  # 参数初始化
            if isinstance(m, nn.Conv2d):  # 判断是否为已知类型
                # kaiming_normal初始化   针对于ReLu激活函数何恺明提出的已知初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 判断是否为已知类型
                # 使用固定值初始化
                nn.init.constant_(m.weight, 1)  # 全1
                nn.init.constant_(m.bias, 0)  # 全0

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            # 如果是全零初始化
            for m in self.modules():
                if isinstance(m, Bottleneck):  # 判断是否为已知类型
                    nn.init.constant_(m.bn3.weight, 0)  # 全0
                elif isinstance(m, BasicBlock):  # 判断是否为已知类型
                    nn.init.constant_(m.bn2.weight, 0)  # 全0

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        '''
        用于生成不同类型的层
        :param block: ResNet网络名称  残差块
        :param planes: 输出通道
        :param blocks: 残差块数量
        :param stride: 卷积层步长
        :param dilate: 是否扩展
        :return:
        '''
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            # 空洞卷积
            self.dilation *= stride  # 计算膨胀大小
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            # 判断步长是否为1，判断  残差块的输入深度 x 残差块卷积层深度  是否等于残差块的扩张
            # 否则添加下采样层
            downsample = nn.Sequential(
                # 一旦判断条件成立，那么给downsample赋予一层1*1卷积层和一层归一化层    使特征图缩小1/2

                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # block()生成上面定义的基础块和瓶颈块的对象，并将dowsample传递给block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        # 改变后面的残差块的输入深度
        # 使得该阶段下面blocks-1个block，即下面循环内构造的block与下一阶段的第一个block的在输入深度上是相同的。
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):  # 将每个block添加到网络
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        # 将layers中的所有block按顺序接在一起并返回
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        #identity = x
        #x = self.ca(x) * x
        #x = self.sa(x) * x
        #x+=identity

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        identity = x
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x = x+identity
        

        
        x = self.avgpool(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    # 用于生成ResNeXt网络
    model = ResNet(block, layers, **kwargs)
    state_dict = model.state_dict()
    if pretrained:  # 加载预训练模型
        state_dict_pro = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict.update(state_dict_pro)
        model.load_state_dict(state_dict)

    return model


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    # 用于生成ResNet网络
    model = ResNet(block, layers, **kwargs)
    if pretrained:  # 加载预训练模型
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnext50_32x4d_attention(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnext('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                    pretrained, progress, **kwargs)


def resnext101_32x8d_attention(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8

    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                    pretrained, progress, **kwargs)


# def Resnext101_32x16d_wsl_attention(progress=True, **kwargs):
# kwargs['groups'] = 32
# kwargs['width_per_group'] = 16
# return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def resnext101_32x16d_wsl_attention(pretrained=False,progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnext101_32x32d_wsl_attention(pretrained=False,progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 32
    return _resnext('resnext101_32x32d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnext101_32x48d_wsl_attention(pretrained=False,progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 48
    return _resnext('resnext101_32x48d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

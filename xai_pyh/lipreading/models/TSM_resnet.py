
import math
import torch.nn as nn
#import pdb
import torch



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

def downsample_basic_block_v2( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )


def threeD_to_2D_tensor(x):
    n_batch, s_time, n_channels, sx, sy = x.shape
    #x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)

def shift(x, n_segment, fold_div=3, inplace=False):
    nt, c, h, w = x.size()
    
    if nt % n_segment != 0:
        n_segment = 4
        if nt % n_segment != 0:
            n_segment = 3
        
    n_t = nt // n_segment
    x = x.view(n_segment, n_t, c, h, w)
    #print("---shift---")
    #print("shift shape",x.shape)
    fold = c // fold_div
    if inplace:
        # Due to some out of order error when performing parallel computing. 
        # May need to write a CUDA kernel.
        raise NotImplementedError  
        # out = InplaceShift.apply(x, fold)
    else:
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        #print("shift start")
        #print(out)
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        #print(out)
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        #print(out)
        #print("---merge---") 
    return out.view(nt, c, h, w)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type = 'relu' ):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu','prelu']

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        # type of ReLU is an input option
        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception('relu type not implemented')
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
    

    def forward(self, x):
        #residual = threeD_to_2D_tensor(x)   # residual shape = [B*T,C,H,W]
        residual = x
        out = shift(x, n_segment=batch, fold_div=16,inplace=False)
        out = self.conv1(out)
        #print(1)
        out = self.bn1(out)
        #print(2)
        out = self.relu1(out)
        out = self.conv2(out)
        #print(3)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual ##
        out = self.relu2(out)
        
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, relu_type = 'relu', gamma_zero = False, avg_pool_downsample = False):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #nn.init.ones_(m.weight)
                #nn.init.zeros_(m.bias)

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock ):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):


        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block( inplanes = self.inplanes, 
                                                 outplanes = planes * block.expansion, 
                                                 stride = stride )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, relu_type = self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type = self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x, B):
        global batch 
        batch = B
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

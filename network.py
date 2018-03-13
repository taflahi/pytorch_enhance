import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from torchvision.models import vgg

import numpy as np

# This is Neural Enhance on PyTorch


class BasicLayer(nn.Module):

    def __init__(self, input_channel, output_channel, kernel=3, stride=1, pad=1, alpha=0.25):
        super(BasicLayer, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel, stride=stride, padding=pad)
        self.prelu = nn.PReLU(init=alpha)

    def forward(self, input):
        x = self.conv(input)
        return self.prelu(x)


class ResidualBlockLayer(nn.Module):

    def __init__(self, input_channel):
        super(ResidualBlockLayer, self).__init__()
        self.basic_layer = BasicLayer(
            input_channel, input_channel, 3, 1, 1, 0.1)

    def forward(self, input):
        x = self.basic_layer(input)
        return torch.add(input, x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.residual_size = 4
        
        # init
        self.init_layer = BasicLayer(3, 64, kernel=7, pad=3)
        
        # residual layers
        self.block_layer = []
        for i in range(self.residual_size):
            self.block_layer.append(ResidualBlockLayer(64))
        
        # upscale layers
        self.upscale_layer = BasicLayer(64, 64 * 4)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        self.output_layer = nn.Conv2d(64, 3, 7, padding=3)
    
    def forward(self, input):
        x = self.init_layer(input)
        for i in range(self.residual_size):
            x = self.block_layer[i](x)
        
        x = self.upscale_layer(x)
        x = self.pixel_shuffle(x)
        return self.output_layer(x)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        
        self.lambd = lambd
    
    def forward(self, input):
        return self.lambd(input)

class Perceptual(nn.Module):
    def __init__(self):
        super(Perceptual, self).__init__()
        
        offset_ = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,3,1,1))
        self.offset = autograd.Variable(torch.from_numpy(offset_), requires_grad=False)
        self.lambd = lambd = LambdaLayer(lambda x: ((x+0.5)*255.0) - self.offset)
        
        # init with pretrained vgg19
        original = vgg.vgg19(pretrained=True)
        self.features = list(original.features.children())[:32]
    
    def forward(self, input):
        conv_1_2, conv_2_2, conv_3_2 = None, None, None
        x = self.lambd(input)
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i == 3:
                conv_1_2 = x.clone()
            elif i == 7:
                conv_2_2 = x.clone()
            elif i == 11:
                conv_3_2 = x.clone()
        
        return conv_1_2, conv_2_2, conv_3_2, x

class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        
        self.channels = channels
        self.input_channel = 64
        
        self.batch_norm1 = nn.BatchNorm2d(self.input_channel)
        self.conv_layer1_1 = BasicLayer(self.input_channel, self.channels, 5, 2, 2)
        self.conv_layer1_2 = BasicLayer(self.channels, self.channels, 5, 2, 2)
        
        self.batch_norm2 = nn.BatchNorm2d(2 * self.input_channel)
        self.conv_layer2 = BasicLayer(2 * self.input_channel, 2 * self.channels, 5, 2, 2)
        
        self.batch_norm3 = nn.BatchNorm2d(4 * self.input_channel)
        self.conv_layer3 = BasicLayer(4 * self.input_channel, 3 * self.channels, 3, 1, 1)
        
        self.conv_layer4 = BasicLayer(3 * self.input_channel, 4 * self.channels, 1, 1, 0)
        self.conv_layer5 = BasicLayer(4 * self.channels, 3 * self.channels, 3, stride=2)
        self.conv_layer6 = BasicLayer(3 * self.channels, 2 * self.channels, 1, 1, 0)
        
        self.batch_norm7 = nn.BatchNorm2d(2 * self.channels)
        self.conv_layer7 = nn.Conv2d(2 * self.channels, 1, 1)
    
    def forward(self, conv_1_2, conv_2_2, conv_3_2):
        x1 = self.batch_norm1(conv_1_2)
        x1 = self.conv_layer1_1(x1)
        x1 = self.conv_layer1_2(x1)
        
        x2 = self.batch_norm2(conv_2_2)
        x2 = self.conv_layer2(x2) 
        
        x3 = self.batch_norm3(conv_3_2)
        x3 = self.conv_layer3(x3) 
        x = torch.cat([x1, x2, x3], dim=1)
        
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        
        x = self.batch_norm7(x)
        return self.conv_layer7(x)

def mse_loss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()

def loss_perceptual(input, target):
    return mse_loss(input, target)

def loss_total_variation(x):
    return torch.mean(((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25)

def softminus(x):
    return x - F.softplus(x)

def loss_adversarial(input):
    return torch.mean(1 - softminus(input))

def loss_discriminator(input, target):
    return torch.mean(softminus(input) - F.softplus(target))

# decay learning rate
def update_optimizer_lr(optimizer, l_r):
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = l_r

def decay_learning_rate(basic_lr, lr_period, lr_decay):
    l_r, t_cur = basic_lr, 0

    while True:
        yield l_r
        t_cur += 1
        if t_cur % lr_period == 0: l_r *= lr_decay

# construct full network
class Enhancer(nn.Module):
    def __init__(self, discriminator_size=32):
        super(Enhancer, self).__init__()
        
        self.generator = Generator()
        self.perceptual = Perceptual()
        self.discriminator = Discriminator(discriminator_size)
    
    def create_new_discriminator(self, size):
        self.discriminator = Discriminator(size)
    
    def forward(self, inputs, seeds):
        inputs = autograd.Variable(torch.from_numpy(inputs))
        seeds = autograd.Variable(torch.from_numpy(seeds))
        
        gen_out = self.generator(seeds)
        c12, c22, c32, perc_out = self.perceptual(torch.cat([inputs, gen_out], dim=0))
        disc_out = self.discriminator(c12, c22, c32)
        
        return gen_out, c12, c22, c32, perc_out, disc_out
    
    def discriminator_clone(self):
        disc = Discriminator(self.discriminator.channels)

        mp = list(disc.parameters())
        mcp = list(self.discriminator.parameters())
        n = len(mp)
        for i in range(0, n):
            mp[i].data[:] = mcp[i].data[:]
            
        return disc
    
    def assign_back_discriminator(self, disc):
        mp = list(self.discriminator.parameters())
        mcp = list(disc.parameters())
        n = len(mp)
        for i in range(0, n):
            mp[i].data[:] = mcp[i].data[:]



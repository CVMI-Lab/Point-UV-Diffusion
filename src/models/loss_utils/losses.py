import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchvision.transforms as T

class HighPassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.blurrer = T.GaussianBlur(kernel_size=(5, 5), sigma=3)

    def forward(self, x, target):
        # Apply a Gaussian blur to the output and target images
        blurred_x = self.blurrer(x)
        blurred_target = self.blurrer(target)

        # Subtract the blurred images from the original images to obtain the high-pass filtered images
        highpass_x = x - blurred_x
        highpass_target = target - blurred_target

        # Compute the L2 distance between the high-pass filtered images
        loss = nn.functional.mse_loss(highpass_x, highpass_target)

        return loss

class PerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False, feature_layers=[0, 1], input_normalize=False):
        super(PerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).cuda()
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).float()
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).float()
        self.resize = resize
        self.input_normalize = input_normalize
        self.feature_layers = feature_layers

    def forward(self, input, target, style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        if self.input_normalize:
            input = (1+input)/2.0
            target = (1+target)/2.0

        input = (input-self.mean.to(input.device)) / self.std.to(input.device)
        target = (target-self.mean.to(input.device)) / self.std.to(input.device)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in self.feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class L2Loss:
    def __call__(self, inputs, targets):
        return torch.nn.functional.mse_loss(inputs, targets)

class L1Loss:
    def __call__(self, inputs, targets):
        return torch.nn.functional.l1_loss(inputs, targets)



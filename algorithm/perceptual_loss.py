import torchvision
from torch import nn


class Perc(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.model = VGG19(
            requires_grad=False
        )
        self.criterion = nn.MSELoss()

    def forward(self, pred_img, gt_img):
        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)
        loss = 0
        for i in range(0, len(gt_fs)):
            loss += self.criterion(pred_fs[i], gt_fs[i])
        return loss


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True
        ).features
        self.slice1 = nn.Sequential()

        for x in range(0, 14):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        out = [h_relu1]
        return out

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision
import timm
import kornia
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse


class NormalizedModel(torch.nn.Module):

    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.nn.Parameter(torch.Tensor(mean).view(-1, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.Tensor(std).view(-1, 1, 1), requires_grad=False)

    def forward(self, x):
        out = (x - self.mean) / self.std 
        out = self.model(out)
        return out


def main(args):
    
    device = args.device
    
    dataset = torchvision.datasets.ImageNet(args.imagenet, 
                                            split="val", 
                                            transform=torchvision.transforms.Compose(
                                                [
                                                    torchvision.transforms.Resize(256), 
                                                    torchvision.transforms.CenterCrop(224), 
                                                    torchvision.transforms.ToTensor()
                                                ]
                                            )
                                           )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=1)

    
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    #model = NormalizedModel(model, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model = timm.create_model(args.model, pretrained=True)
    model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in tqdm(dataloader):
            bx = x.to(device)
            by = y.to(device)

            a = torch.empty(len(by))
            is_correct = torch.ones(len(by)).bool().to(device)

            for _ in range(args.iterations):
                a.data[is_correct] = torch.tensor(np.random.uniform(0, 1, len(a.data[is_correct]))).float()

                x_aug = kornia.enhance.solarize(bx, a)
                logits = model(x_aug)
                is_correct.data = (logits.argmax(dim=1) == by).detach()
                acc = is_correct.float().mean().item()

            correct += is_correct.float().sum().item()
            total += len(by)
        
    print(correct / total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    args = parser.parse_args()

    main(args)
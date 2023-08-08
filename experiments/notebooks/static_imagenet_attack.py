import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision
import timm
import kornia
from torch.utils.data import DataLoader
from tqdm import tqdm
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

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)

    device = args.device

    model_name = args.model

    model = timm.create_model(model_name, pretrained=True)
    model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    accs = []

    with torch.no_grad():
        for alpha in tqdm(np.arange(0, 1.00001, 0.01)):
            correct = 0
            total = 0
            for x, y in (dataloader):
                x = x.to(device)
                y = y.to(device)

                x_aug = kornia.enhance.solarize(x, torch.tensor(alpha).float().to(device))

                logits = model(x_aug)
                correct += (logits.argmax(dim=1) == y).float().sum().item()
                total += len(y)

            acc = correct / total
            accs.append(acc)

    np.save(f"output/static_imagenet_{model_name}.npy", np.array(accs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    args = parser.parse_args()

    main(args)
import torch
import timm
from torch.utils.data import DataLoader
import torchvision


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


def get_normalized_model(model_name):
    model = None
    if model_name == "alexnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        model = NormalizedModel(model, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        model = timm.create_model(model_name, pretrained=True)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    model.eval()
    return model


def get_imagenet_loader(path, batch_size, num_workers):
    dataset = torchvision.datasets.ImageNet(path, 
                                            split="val", 
                                            transform=torchvision.transforms.Compose(
                                                [
                                                    torchvision.transforms.Resize(256), 
                                                    torchvision.transforms.CenterCrop(224), 
                                                    torchvision.transforms.ToTensor()
                                                ]
                                            )
                                           )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return dataloader
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
    elif model_name == "robust_resnet50_linf_eps4":
        model = timm.create_model("resnet50", pretrained=False)
        # get torch state from url
        state = torch.hub.load_state_dict_from_url("https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps4.0.ckpt", map_location="cpu")["model"]
        state = {k.replace("module.model.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = NormalizedModel(model, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif model_name == "resnet50_noisymix":
        # grab the file from https://drive.google.com/file/d/1Na79fzPZ0Azg01h6kGn1Xu5NoWOElSuG/ and throw it into your torchhub cache dir - I won't mess with GDrive API
        model = timm.create_model("resnet50", pretrained=True)
        state = torch.hub.load_state_dict_from_url("https://download-it-yourself.com/Erichson2022NoisyMix_new.pt", map_location="cpu")["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))
    else:
        model = timm.create_model(model_name, pretrained=True)
        model = NormalizedModel(model, model.default_cfg.get("mean"), model.default_cfg.get("std"))

    model.eval()
    return model


def get_imagenet_loader(path, batch_size, num_workers, shuffle=False):
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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return dataloader


# Source: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L420
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed):
    torch.manual_seed(seed)

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "y")
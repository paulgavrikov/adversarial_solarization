import torch
from torch.utils.data import DataLoader
import torchvision
import requests
from model_zoo import get_normalized_model


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


def get_gpu_stats():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    
    nvmlInit()
    stats = []
    for i in range(torch.cuda.device_count()):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        stats.append(info.used)
    return stats


def autoselect_device():
    best_is_gpu = False

    device = "cpu"

    try:
        mps_available = torch.backends.mps.is_available()
    except:
        mps_available = False

    if torch.cuda.is_available():
        best_is_gpu = True
    elif mps_available:
        device = "mps"
    else:
        device = "cpu"


    if best_is_gpu:
        import numpy as np

        best_device = f"cuda:{np.argmin(get_gpu_stats())}"
        device = best_device

    return device


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

import torch
import torch.utils.data
from tqdm import tqdm
import argparse
import torchvision
from torch.utils.data import DataLoader
from utils import get_normalized_model
import os
try:
    import wandb
    HAS_WANDB = True
except:
    HAS_WANDB = False


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', # noise
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', # blur
    'snow', 'frost', 'fog', 'brightness', # weather
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', # digital
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate' # extra
]

SEVERITIES = [1, 2, 3, 4, 5]


def get_loader(path, batch_size, num_workers):
    dataset = torchvision.datasets.ImageFolder(path, 
                                                transform=torchvision.transforms.Compose(
                                                    [
                                                        torchvision.transforms.Resize(256), # Hendrycks et al. did not resize
                                                        torchvision.transforms.CenterCrop(224), 
                                                        torchvision.transforms.ToTensor()
                                                    ]
                                                )
                                            )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return dataloader


def main(args):

    run = None
    if HAS_WANDB:
        run = wandb.init(project="imagenet_c", name=args.model, config=args)

    device = args.device
    
    model = get_normalized_model(args.model)
    model.to(device)

    accs = []

    tests = 0
    
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:

            tests += 1

            c_path = os.path.join(args.imagenet, f"{corruption}/{severity}/")

            dataloader = get_loader(path=c_path, batch_size=args.batch_size, num_workers=16)

            correct = 0
            total = 0

            with torch.no_grad():

                for x, y in tqdm(dataloader, desc=f"({tests}/{len(CORRUPTIONS) * len(SEVERITIES)}) {corruption}/{severity}"):
                    bx = x.to(device)
                    by = y.to(device)

                    # assert that bx is not normalized by mean and std
                    assert torch.all(bx >= 0) and torch.all(bx <= 1), "Data must be in [0, 1] range"

                    logits = model(bx)
                    is_correct = (logits.argmax(dim=1) == by).detach()

                    correct += is_correct.float().sum().item()
                    total += len(by)

            acc = (correct / total) * 100
            accs.append(acc)
            
            print(f"{corruption}/{severity} - {acc:.2f}")
            if run:
                run.log({f"{corruption}/{severity}": acc})

    mca = sum(accs) / len(accs)

    print(f"Average Accuracy: {mca:.2f}")
    if run:
        run.log({f"mean": mca})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    args = parser.parse_args()

    main(args)

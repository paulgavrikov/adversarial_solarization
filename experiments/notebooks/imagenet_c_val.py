import torch
import torch.utils.data
from tqdm import tqdm
import argparse
import torchvision
from torch.utils.data import DataLoader
from utils import get_normalized_model, accuracy, AverageMeter, str2bool
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
    if HAS_WANDB and args.wandb:
        run = wandb.init(project="imagenet_c", name=args.model, config=args)

    device = args.device
    
    model = get_normalized_model(args.model)
    model.to(device)

    global_top1_meter = AverageMeter()
    global_top5_meter = AverageMeter()

    tests = 0
    
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:

            tests += 1

            c_path = os.path.join(args.imagenet, f"{corruption}/{severity}/")

            dataloader = get_loader(path=c_path, batch_size=args.batch_size, num_workers=args.num_workers)

            top1_meter = AverageMeter()
            top5_meter = AverageMeter()

            with torch.no_grad():

                for x, y in tqdm(dataloader, desc=f"({tests}/{len(CORRUPTIONS) * len(SEVERITIES)}) {corruption}/{severity}"):
                    bx = x.to(device)
                    by = y.to(device)

                    # assert that bx is not normalized by mean and std
                    assert torch.all(bx >= 0) and torch.all(bx <= 1), "Data must be in [0, 1] range"

                    logits = model(bx)
                    
                    top1, top5 = accuracy(logits, by, topk=(1, 5))
                    top1_meter.update(top1.item(), bx.size(0))
                    top5_meter.update(top5.item(), bx.size(0))

            global_top1_meter.update(top1_meter.avg)
            global_top5_meter.update(top5_meter.avg)
            
            print(f"{corruption}/{severity} - top1: {top1_meter.avg:.2f}%, top5: {top5_meter.avg:.2f}%")
            if run:
                run.log(
                    {
                        f"top1_acc/{corruption}_{severity}": top1_meter.avg,
                        f"top5_acc/{corruption}_{severity}": top5_meter.avg,
                    }
                )

    print(f"Average Accuracy top1: {global_top1_meter.avg:.2f}%, top5: {global_top5_meter.avg:.2f}%")
    if run:
        run.log({f"top1_mean": global_top1_meter.avg, "top5_mean": global_top5_meter.avg})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb', type=str2bool, default=True)
    args = parser.parse_args()

    main(args)

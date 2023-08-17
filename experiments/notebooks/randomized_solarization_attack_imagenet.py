import numpy as np
import torch
import torch.utils.data
import kornia
from tqdm import tqdm
import numpy as np
import argparse
from utils import get_normalized_model, get_imagenet_loader, accuracy, AverageMeter


def main(args):
    
    device = args.device
    
    dataloader = get_imagenet_loader(path=args.imagenet, batch_size=args.batch_size, num_workers=16)

    model = get_normalized_model(args.model)
    model.to(device)

    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    with torch.no_grad():

        for x, y in tqdm(dataloader):
            bx = x.to(device)
            by = y.to(device)

            # assert that bx is not normalized by mean and std
            assert torch.all(bx >= 0) and torch.all(bx <= 1), "Data must be in [0, 1] range"

            a = torch.empty(len(by))
            is_correct = torch.ones(len(by)).bool().to(device)

            for _ in range(args.iterations):
                a.data[is_correct] = torch.tensor(np.random.uniform(0, 1, len(a.data[is_correct]))).float()

                x_aug = kornia.enhance.solarize(bx, a)
                logits = model(x_aug)
                is_correct.data = (logits.argmax(dim=1) == by).detach()

            top1, top5 = accuracy(logits, by, topk=(1, 5))
            top1_meter.update(top1.item(), bx.size(0))
            top5_meter.update(top5.item(), bx.size(0))
        
    print(f"Robust accuracy top1: {top1_meter.avg * 100:.2f} %, top5: {top5_meter.avg * 100:.2f} %")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    args = parser.parse_args()

    main(args)
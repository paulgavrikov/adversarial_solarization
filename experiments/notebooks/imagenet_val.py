import torch
import torch.utils.data
from tqdm import tqdm
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

            logits = model(bx)
            
            top1, top5 = accuracy(logits, by, topk=(1, 5))
            top1_meter.update(top1.item(), bx.size(0))
            top5_meter.update(top5.item(), bx.size(0))

    print(f"Accuracy top1: {top1_meter.avg:.2f}%, top5: {top5_meter.avg:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    args = parser.parse_args()

    main(args)

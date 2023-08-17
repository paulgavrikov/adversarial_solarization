import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from utils import get_normalized_model, get_imagenet_loader, accuracy, AverageMeter


def main(args):

    device = args.device
    
    dataloader = get_imagenet_loader(path=args.imagenet, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = get_normalized_model(args.model)
    model.to(device)

    top1_meter = AverageMeter()
    top5_meter = AverageMeter()


    for x, y in tqdm(dataloader):
        bx = x.to(device)
        by = y.to(device)

        p = torch.zeros_like(bx, requires_grad=True, device=device)

        criterion = torch.nn.CrossEntropyLoss()

        adv_logits = model(bx + p)
        adv_loss = criterion(adv_logits, by)
        adv_loss.backward()

        p.data = p.data + args.eps * p.grad.data.sign()  # gradient ascent!
        p.grad.data.zero_()

        x_adv = torch.clip(bx + p, 0, 1).detach()

        with torch.no_grad():
            logits = model(x_adv)
            
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
    parser.add_argument('--num_workers', type=int, default=8)

    # attack parameter
    parser.add_argument('--eps', type=float, default=4/255)

    args = parser.parse_args()

    main(args)

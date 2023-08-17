import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from utils import get_normalized_model, get_imagenet_loader, accuracy, AverageMeter


def ce_loss(logits, labels):
    """
    Cross-entropy loss
    """
    probs = torch.softmax(logits, dim=1)
    loss = probs.gather(1, labels.unsqueeze(1)).squeeze(1).log().neg().mean()
    return loss


def cw_loss(logits, labels):
    """
    Carlini-Wagner loss
    """
    probs = torch.softmax(logits, dim=1)

    onehot_labels = torch.eye(len(probs[0]))[labels].to(probs.device)

    real = (onehot_labels * probs).sum(1)
    other = ((1.0 - onehot_labels) * probs - onehot_labels * 10000.0).max(1)[0]

    loss = torch.clamp(other - real + 50.0, min=0.0).mean()
    return loss


def fgsm_attack(model, bx, by, eps, criterion):
    """
    Fast Gradient Sign Method
    """
    p = torch.zeros_like(bx, requires_grad=True)

    adv_logits = model(bx + p)
    adv_loss = criterion(adv_logits, by)
    adv_loss.backward()

    p.data = p.data + eps * p.grad.data.sign()  # gradient ascent!
    p.grad.data.zero_()

    return torch.clip(bx + p, 0, 1).detach()


def main(args):

    device = args.device

    criterion = None
    if args.loss == "ce":
        criterion = ce_loss
    elif args.loss == "cw":
        criterion = cw_loss
    else:
        raise ValueError(f"Unknown loss {args.loss}")
    
    dataloader = get_imagenet_loader(path=args.imagenet, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = get_normalized_model(args.model)
    model.to(device)

    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for x, y in tqdm(dataloader):
        bx = x.to(device)
        by = y.to(device)

        x_adv = fgsm_attack(model, bx, by, args.eps, criterion)

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
    parser.add_argument('--loss', type=str, default="ce", choices=["ce", "cw"])

    args = parser.parse_args()

    main(args)

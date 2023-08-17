import numpy as np
import torch
import torch.utils.data
import kornia
from tqdm import tqdm
import numpy as np
import argparse
from utils import get_normalized_model, get_imagenet_loader, accuracy, seed_everything, AverageMeter


def rand_sol_attack(model, bx, by, iterations, target, device):
    """
    Randomized Solarization Attack

    :param model: model to attack
    :param bx: batch of images
    :param by: batch of labels
    :param iterations: number of iterations of the attack
    :param target: target of the attack, either "top1", "top5", or "ce_loss"
    :param device: device to use
    :return: logits of the final attack, parameters of the final attack
    """
    params = torch.empty(len(by))
    is_correct = torch.ones(len(by)).bool().to(device)

    for _ in range(iterations):
        params.data[is_correct] = torch.tensor(np.random.uniform(0, 1, len(params.data[is_correct]))).float()

        x_aug = kornia.enhance.solarize(bx, params)
        logits = model(x_aug)

        if target == "top1":
            is_correct.data = (logits.argmax(dim=1) == by).detach()
        else:
            raise NotImplementedError("Only top1 is currently supported")
    return logits, params


def main(args):

    seed_everything(args.seed)
    
    device = args.device
    
    dataloader = get_imagenet_loader(path=args.imagenet, batch_size=args.batch_size, num_workers=16, shuffle=False)

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

            final_logits, _ = rand_sol_attack(model, bx, by, args.iterations, args.target, device)

            top1, top5 = accuracy(final_logits, by, topk=(1, 5))
            top1_meter.update(top1.item(), bx.size(0))
            top5_meter.update(top5.item(), bx.size(0))
        
    print(f"Robust accuracy top1: {top1_meter.avg:.2f}%, top5: {top5_meter.avg:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    parser.add_argument('--target', type=str, choices=["top1", "top5", "ce_loss"], default="top1")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args)
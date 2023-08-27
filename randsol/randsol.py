import numpy as np
import torch
import torch.utils.data
import kornia
from tqdm import tqdm
import numpy as np
import argparse
from utils import get_normalized_model, get_imagenet_loader, accuracy, seed_everything, autoselect_device, str2bool, AverageMeter

try:
    import wandb
    HAS_WANDB = True
except:
    HAS_WANDB = False


def rand_sol_attack(model, bx, by, iterations, target):
    """
    Randomized Solarization Attack

    :param model: model to attack
    :param bx: batch of images
    :param by: batch of labels
    :param iterations: number of iterations of the attack
    :param target: target of the attack "top{k}" for any k > 0
    :return: logits of the final attack, parameters of the final attack
    """
    assert iterations > 0, "Number of iterations must be greater than 0"
    assert len(bx) == len(by), "Batch size of bx and by must be equal"

    if target.startswith("top") :
        k = int(target[3:])
        return _rand_sol_attack_accuracy(model=model, bx=bx, by=by, iterations=iterations, k=k)
    else:
        raise NotImplementedError(f"{target} not supported")


def _rand_sol_attack_accuracy(model, bx, by, iterations, k):
    params = torch.empty(len(by))
    is_correct = torch.ones(len(by)).bool()

    for _ in range(iterations):
        params.data[is_correct] = torch.tensor(np.random.uniform(0, 1, len(params.data[is_correct]))).float()

        x_aug = kornia.enhance.solarize(bx, params)
        logits = model(x_aug)

        _, pred = logits.cpu().topk(k, 1, True, True)
        pred = pred.t()
        k_correct = pred.eq(by.cpu().view(1, -1).expand_as(pred))

        is_correct.data = k_correct[:k].float().max(0).values.bool().detach()
        
    return logits, params


def main(args):

    run = None
    if HAS_WANDB and args.wandb:
        run = wandb.init(project="RandSol", name=args.model, config=args)

    seed_everything(args.seed)
    
    device = args.device if args.device is not None else autoselect_device()

    print(f"Using device {device}")
    
    dataloader = get_imagenet_loader(path=args.imagenet, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = get_normalized_model(args.model)
    model.to(device)

    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    with torch.no_grad():

        progress = tqdm(dataloader, desc=f"RandSol")

        for x, y in progress:
            bx = x.to(device)
            by = y.to(device)

            # assert that bx is not normalized by mean and std
            assert torch.all(bx >= 0) and torch.all(bx <= 1), "Data must be in [0, 1] range"

            final_logits, _ = rand_sol_attack(model=model, bx=bx, by=by, iterations=args.iterations, target=args.target)

            top1, top5 = accuracy(final_logits, by, topk=(1, 5))
            top1_meter.update(top1.item(), bx.size(0))
            top5_meter.update(top5.item(), bx.size(0))

            progress.set_postfix({"top1": top1_meter.avg, "top5": top5_meter.avg})
        
    print(f"Robust accuracy top1: {top1_meter.avg:.2f}%, top5: {top5_meter.avg:.2f}%")
    if run:
        run.log({f"acc/top1": top1_meter.avg, "acc/top5": top5_meter.avg})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--wandb', type=str2bool, default=True)

    # attack parameters
    parser.add_argument('--target', type=str, default="top1")
    parser.add_argument('--iterations', type=int, default=10)
    
    args = parser.parse_args()

    main(args)
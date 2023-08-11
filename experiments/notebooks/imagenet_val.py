import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from utils import get_normalized_model, get_imagenet_loader


def main(args):

    device = args.device
    
    dataloader = get_imagenet_loader(path=args.imagenet, batch_size=args.batch_size, num_workers=16)

    model = get_normalized_model(args.model)
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in tqdm(dataloader):
            bx = x.to(device)
            by = y.to(device)

            logits = model(bx)
            is_correct = (logits.argmax(dim=1) == by).detach()

            correct += is_correct.float().sum().item()
            total += len(by)

    print(f"{(correct / total) * 100:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    args = parser.parse_args()

    main(args)

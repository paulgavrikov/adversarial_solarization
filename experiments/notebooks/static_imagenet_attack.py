import numpy as np
import torch
import torch.utils.data
import kornia
from tqdm import tqdm
import argparse
from utils import get_normalized_model, get_imagenet_loader


def main(args):
    dataloader = get_imagenet_loader(path=args.imagenet, batch_size=args.batch_size, num_workers=16)

    device = args.device

    model = get_normalized_model(args.model)    
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    accs = []

    with torch.no_grad():
        for alpha in tqdm(np.arange(0, 1.00001, 0.01)):
            correct = 0
            total = 0
            for x, y in (dataloader):
                x = x.to(device)
                y = y.to(device)

                # assert that bx is not normalized by mean and std
                assert torch.all(x >= 0) and torch.all(x <= 1), "Data must be in [0, 1] range"

                x_aug = kornia.enhance.solarize(x, torch.tensor(alpha).float().to(device))

                logits = model(x_aug)
                correct += (logits.argmax(dim=1) == y).float().sum().item()
                total += len(y)

            acc = correct / total
            accs.append(acc)

    np.save(f"output/static_imagenet_{args.model}.npy", np.array(accs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    args = parser.parse_args()

    main(args)
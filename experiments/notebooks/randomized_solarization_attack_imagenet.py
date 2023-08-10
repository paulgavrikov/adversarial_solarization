import numpy as np
import torch
import torch.utils.data
import kornia
from tqdm import tqdm
import numpy as np
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

            # assert that bx is not normalized by mean and std
            assert torch.all(bx >= 0) and torch.all(bx <= 1), "Data must be in [0, 1] range"

            a = torch.empty(len(by))
            is_correct = torch.ones(len(by)).bool().to(device)

            for _ in range(args.iterations):
                a.data[is_correct] = torch.tensor(np.random.uniform(0, 1, len(a.data[is_correct]))).float()

                x_aug = kornia.enhance.solarize(bx, a)
                logits = model(x_aug)
                is_correct.data = (logits.argmax(dim=1) == by).detach()

            correct += is_correct.float().sum().item()
            total += len(by)
        
    print(f"{(correct / total) * 100:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--imagenet', type=str, default="/home/SSD/ImageNet/")
    args = parser.parse_args()

    main(args)
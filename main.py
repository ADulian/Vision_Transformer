import argparse

import torch

from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from vit import Vision_Transformer

# --------------------------------------------------------------------------------
def main(args):

    # Params
    batch_size = args.batch_size
    img_size = args.img_size
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    # Data
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model
    net = Vision_Transformer(img_size=img_size,
                             in_channels=3,
                             num_layers=args.num_layers,
                             patch_size=args.patch_size,
                             embedding_dim=args.emb_d,
                             forward_expansion=args.forward_expansion,
                             lr=args.lr,
                             )

    # Training
    net.fit(train_loader=train_loader)

# --------------------------------------------------------------------------------
if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser()

    # Args
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_d', type=float, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--forward_expansion', type=int, default=4)
    parser.add_argument('--patch_size', type=float, default=16)

    args = parser.parse_args()

    main(args=args)
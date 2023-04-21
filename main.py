import argparse

from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from vit import Vision_Transformer

# --------------------------------------------------------------------------------
def main(args):
    batch_size = args.batch_size
    img_size = args.img_size

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
                             patch_size=args.patch_size,
                             patch_embedding_dim=args.patch_emb,
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
    parser.add_argument('--patch_emb', type=float, default=512)
    parser.add_argument('--patch_size', type=float, default=16)

    args = parser.parse_args()

    main(args=args)
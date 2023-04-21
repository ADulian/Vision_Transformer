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
    net = Vision_Transformer()

    # Training
    net.fit(train_loader=train_loader)

# --------------------------------------------------------------------------------
if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser()

    # Args
    parser.add_argument('--batch_size', '-bs', type=int, required=False, default=2)
    parser.add_argument('--img_size', '-is', type=int, required=False, default=32)

    args = parser.parse_args()

    main(args=args)
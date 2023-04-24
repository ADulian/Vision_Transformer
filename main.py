import argparse

import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vit import Vision_Transformer

# --------------------------------------------------------------------------------
def main(args):

    # Params
    batch_size = args.batch_size
    img_size = args.img_size
    device = torch.device("cuda") if (torch.cuda.is_available() and args.cuda) else torch.device("cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    # Data
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              pin_memory=args.pin_memory, num_workers=args.num_workers, shuffle=True)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                             pin_memory=args.pin_memory, num_workers=args.num_workers)

    # Model
    model = Vision_Transformer(device=device,
                               num_classes=len(train_set.classes),
                               img_size=img_size,
                               in_channels=3,
                               num_layers=args.num_layers,
                               patch_size=args.patch_size,
                               embedding_dim=args.emb_d,
                               forward_expansion=args.forward_expansion,
                               lr=args.lr,
                               ).to(device)
    # Device info
    print(f"\nUsing {device} device")

    # Load model if path exists
    if args.load_path:
        print(f"\nLoading model from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path))

    # Training
    model.fit(train_loader=train_loader, epochs=args.epochs)
    model.test(test_loader=test_loader)

    # Save model
    print(f"\nSaving model as {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)

# --------------------------------------------------------------------------------
if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser()

    # Args
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_d', type=float, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--forward_expansion', type=int, default=4)
    parser.add_argument('--patch_size', type=float, default=16)
    parser.add_argument('--save_path', type=str, default="model.pth")
    parser.add_argument('--load_path', type=str, default="")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    main(args=args)
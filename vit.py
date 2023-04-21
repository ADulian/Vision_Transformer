
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------------------------------------
class Vision_Transformer(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self,
                 lr=1e-3):
        super().__init__()

        # Layers
        self.patch_embeddings = Patch_Embedding()

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optim
        self.optim = optim.Adam(self.parameters(), lr=lr)

    # --------------------------------------------------------------------------------
    def forward(self, x):
        out = self.patch_embeddings(x)

        return out

    # --------------------------------------------------------------------------------
    def fit(self, train_loader, epochs=10):
        print("Training...")

        for epoch in range(epochs):

            epoch_loss = 0.0
            for i, data in enumerate(train_loader):
                # Data
                x, y = data

                # Zero grad
                self.optim.zero_grad()

                # Forward
                y_hat = self(x)

                # Loss + Backward
                loss = self.criterion(y_hat, y)
                loss.backward()

                # Step
                self.optim.step()

                # Add loss
                epoch_loss += loss.item()

            # Mean loss
            epoch_loss = epoch_loss / len(train_loader)
            print(f"--- Epoch: {epoch} Loss: {epoch_loss}")


# --------------------------------------------------------------------------------
class Patch_Embedding(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embedding_dimensions: int = 512):

        super().__init__()

        # Attribs
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embedding_dimensions = embedding_dimensions

        # Instead of using Linear Layer a Conv layer with stride and kernel size equal to patch size can be
        # used to get the same results, kernel will essentially look at each patch without any overlaps
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dimensions,
                              kernel_size=patch_size, stride=patch_size)

    # --------------------------------------------------------------------------------
    def forward(self, x):


        # Input shape -> (batch_size, in_channels, height, width)
        batch_size = x.shape[0]

        # Output shape -> (batch_size, out_channels (embedding_dimensions), num_patches ** 0.5, num_patches ** 0.5)
        out = self.conv(x)

        # Reshape the output so that its shape is
        # Output shape -> (batch_size, embedding_dimensions, num_patches)
        out = out.view(batch_size, self.embedding_dimensions, self.num_patches)

        # Swap dimensions so that the final output shape is
        # Output shape -> (batch_size, num_patches, embedding_dimensions)
        out = out.transpose(1, 2)

        return out

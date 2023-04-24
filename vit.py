import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

# --------------------------------------------------------------------------------
class Vision_Transformer(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self,
                 device,
                 num_classes=10,
                 img_size=32,
                 in_channels=3,
                 num_layers=4,
                 num_heads=8,
                 patch_size=16,
                 embedding_dim=512,
                 forward_expansion=4,
                 lr=1e-3):

        super().__init__()

        # Device
        self.device = device

        # Layers + Params
        self.patch_embeddings = Patch_Embedding(img_size=img_size, in_channels=in_channels,
                                                patch_size=patch_size, embedding_dim=embedding_dim)

        self.transformer = nn.ModuleList([
            Transformer_Encoder(embedding_dim=embedding_dim,
                                num_heads=num_heads,
                                forward_expansion=forward_expansion) for _ in range(num_layers)
        ])

        self.mlp_head = nn.Linear(embedding_dim, num_classes) # Logit over classes

        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1 + self.patch_embeddings.num_patches, embedding_dim))

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optim
        self.optim = optim.Adam(self.parameters(), lr=lr)

    # --------------------------------------------------------------------------------
    def forward(self, x):
        batch_size = x.shape[0]

        # Patch Embeddings
        out = self.patch_embeddings(x)

        # Get class embedding and expand it to match x
        class_token = self.class_token.expand(batch_size, -1, -1)

        # Append class embedding to the output of patch embeddings (final set of tokens)
        out = torch.cat((class_token, out), dim=1)

        # Position Embeddings
        out = out + self.position_embeddings

        # Transformer
        for layer in self.transformer:
            out = layer(out)

        # Prediction head
        out = out[:, 0] # Section 3.1 Eq.4
        out = self.mlp_head(out)

        return out

    # --------------------------------------------------------------------------------
    def fit(self, train_loader, epochs=10):
        print("\nTraining...")

        self.train()

        for epoch in range(epochs):

            with tqdm(total=len(train_loader), dynamic_ncols=True, desc=f"Epoch: {epoch + 1}/{epochs}") as t:

                epoch_loss = []
                for i, data in enumerate(train_loader):
                    # Data
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)

                    # Zero grad
                    self.optim.zero_grad()

                    # Forward
                    y_hat = self(x)

                    # Loss + Backward
                    loss = self.criterion(y_hat, y)
                    epoch_loss.append(loss.item())

                    loss.backward()

                    # Step
                    self.optim.step()

                    # Update progress bar
                    t.set_postfix(loss=np.mean(epoch_loss))
                    t.update()

    # --------------------------------------------------------------------------------
    def test(self, test_loader):
        print("\nTesting...")

        self.eval()

        # Get test loss
        test_loss = []
        correct = 0
        with tqdm(total=len(test_loader), dynamic_ncols=True) as t:
            for i, data in enumerate(test_loader):
                with torch.inference_mode(mode=True):
                    # Data
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)

                    # Forward
                    y_hat = self(x)

                    # Loss
                    loss = self.criterion(y_hat, y)
                    test_loss.append(loss.item())

                    # Acc
                    _, predicted = torch.max(y_hat, 1)
                    correct += (predicted == y).sum().item()

                    # Update progress bar
                    t.set_postfix(loss=np.mean(test_loss))
                    t.update()

        print(f"--- Test Loss: {np.mean(test_loss):.4f}")
        print(f"--- Test Acc: {100 * (correct / len(test_loader.dataset)):.2f}%")

# --------------------------------------------------------------------------------
class Patch_Embedding(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embedding_dim: int = 512):

        super().__init__()

        # Attribs
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embedding_dim = embedding_dim

        # --- Layers
        # Instead of using Linear Layer a Conv layer with stride and kernel size equal to patch size can be
        # used to get the same results, kernel will essentially look at each patch without any overlaps
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                              kernel_size=patch_size, stride=patch_size)

    # --------------------------------------------------------------------------------
    def forward(self, x):


        # Input shape -> [batch_size, in_channels, height, width]
        batch_size = x.shape[0]

        # Output shape -> [batch_size, out_channels (embedding_dimensions), num_patches ** 0.5, num_patches ** 0.5]
        out = self.conv(x)

        # Reshape the output so that its shape is
        # Output shape -> [batch_size, embedding_dimensions, num_patches]
        out = out.view(batch_size, self.embedding_dim, self.num_patches)

        # Swap dimensions so that the final output shape is
        # Output shape -> [batch_size, num_patches, embedding_dimensions]
        out = out.transpose(1, 2)

        return out

# --------------------------------------------------------------------------------
class Attention(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, embedding_dim=512, num_heads=8):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert (self.head_dim * num_heads == embedding_dim), "Embedding size needs to be divisible by num of heads"

        # --- Layers
        self.linear_queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear_keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear_values = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.linear_out = nn.Linear(self.embedding_dim, self.embedding_dim)

    # --------------------------------------------------------------------------------
    def forward(self, x): # No mask required for the ViT
        """
        """

        # Batch size
        batch_size = x.shape[0]

        # Number of "tokens" (patches + 1)
        patches_len = x.shape[1]

        # Reshape embedding dimension for Multi-Headed Attention
        # Original shape -> [batch_size, num_patches + 1, embedding_dim]
        # New shape -> [batch_size, num_patches + 1, num_heads, head_dim]
        queries = x.reshape(batch_size, patches_len, self.num_heads, self.head_dim)
        keys = x.reshape(batch_size, patches_len, self.num_heads, self.head_dim)
        values = x.reshape(batch_size, patches_len, self.num_heads, self.head_dim)

        # Linear projection
        # Out shape -> [batch_size, num_patches + 1, num_heads, head_dim]
        queries = self.linear_queries(queries)
        keys = self.linear_keys(keys)
        values = self.linear_values(values)

        # Query @ Keys
        # Out shape -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        similarity = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])

        # Attention Filter
        attention = similarity / (self.head_dim ** (1/2)) # Scale
        attention = torch.softmax(attention, dim=-1) # Squash

        # Attention @ Values
        # Out shape -> [batch_size, num_patches + 1, num_heads, head_dim]
        out = torch.einsum("bhql,blhd->bqhd", [attention, values])

        # Stack Heads
        # Out shape -> [batch_size, num_patches + 1, num_heads * head_dim]
        out = out.reshape(batch_size, patches_len, self.num_heads * self.head_dim)

        # Linear projection
        # Out shape -> [batch_size, num_patches + 1, num_heads * head_dim]
        out = self.linear_out(out)

        return out

# --------------------------------------------------------------------------------
class Transformer_Encoder(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, embedding_dim=512, num_heads=8, forward_expansion=4):
        super().__init__()

        # Norm Layers
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)

        # Attention
        self.attention = Attention(embedding_dim=embedding_dim, num_heads=num_heads)

        # MLP
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * forward_expansion),
                                 nn.GELU(),
                                 nn.Linear(embedding_dim * forward_expansion, embedding_dim))


    # --------------------------------------------------------------------------------
    def forward(self, x):
        """
        """

        # Layer Norm
        # Out Shape -> [batch_size, num_patches + 1, embedding_dim]
        out = self.norm_1(x)

        # Attention
        # Out Shape -> [batch_size, num_patches + 1, embedding_dim]
        out = self.attention(out)

        # Add
        # Out Shape -> [batch_size, num_patches + 1, embedding_dim]
        residuals = out + x

        # Layer Norm
        # Out Shape -> [batch_size, num_patches + 1, embedding_dim]
        out = self.norm_2(residuals)

        # MLP
        # Out Shape -> [batch_size, num_patches + 1, embedding_dim]
        out = self.mlp(out)

        # Add
        # Out Shape -> [batch_size, num_patches + 1, embedding_dim]
        out = out + residuals

        return out

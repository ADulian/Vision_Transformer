
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------------------------------------
class Vision_Transformer(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self,
                 img_size=32,
                 in_channels=3,
                 patch_size=16,
                 patch_embedding_dim=512,
                 lr=1e-3):
        super().__init__()

        # Layers
        self.patch_embeddings = Patch_Embedding(img_size=img_size, in_channels=in_channels,
                                                patch_size=patch_size, embedding_dim=patch_embedding_dim)

        self.attention = Attention()

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optim
        self.optim = optim.Adam(self.parameters(), lr=lr)

    # --------------------------------------------------------------------------------
    def forward(self, x):
        out = self.patch_embeddings(x)
        out = self.attention(out, out, out)

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
                break

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
    def forward(self, queries, keys, values): # No mask required for the ViT
        """
        This could be done with a single input (all inputs are the same anyway), however, this is following
        the original implementation of Transformer which does both self-attention as well as cross-attention
        """

        # Batch size
        batch_size = queries.shape[0]

        # Number of "tokens" (patches + 1)
        queries_len = queries.shape[1]
        keys_len = keys.shape[1]
        values_len = values.shape[1]

        # Reshape embedding dimension for Multi-Headed Attention
        # Original shape -> [batch_size, num_patches + 1, embedding_dim]
        # New shape -> [batch_size, num_patches + 1, num_heads, head_dim]
        queries = queries.reshape(batch_size, queries_len, self.num_heads, self.head_dim)
        keys = queries.reshape(batch_size, keys_len, self.num_heads, self.head_dim)
        values = queries.reshape(batch_size, values_len, self.num_heads, self.head_dim)

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
        out = out.reshape(batch_size, queries_len, self.num_heads * self.head_dim)

        # Linear projection
        # Out shape -> [batch_size, num_patches + 1, num_heads * head_dim]
        out = self.linear_out(out)

        return out



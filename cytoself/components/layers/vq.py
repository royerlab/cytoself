from typing import Optional

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer
    Ref. https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        commitment_cost: float = 0.25,
        padding_idx: Optional[int] = None,
        initializer: str = 'uniform',
        **kwargs,
    ):
        """
        Initializes Vector Quantization layer

        Parameters
        ----------
        embedding_dim : int
            Embedding dimension
        num_embeddings : int
            Number of embeddings
        commitment_cost : float
            Commitment cost
        padding_idx : int
            If specified, the entries at padding_idx do not contribute to the gradient;
            therefore, the embedding vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”.
        initializer : str
            Initializing distribution
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.padding_idx = padding_idx

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim, self.padding_idx)
        if initializer == 'uniform':
            self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def _calc_dist(self, z):
        """
        Computes distance between inputs and codebook.

        Parameters
        ----------
        z : tensor
            Usually the output of encoder

        Returns
        -------
        tensor

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.movedim(z, 1, -1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.codebook.weight.t())
        )
        return distances

    def _calc_metrics(self, z, z_quantized, encoding_onehot, softmax_loss=0):
        """
        Compute metrics and losses

        Parameters
        ----------
        z : tensor
            Usually the output of encoder
        z_quantized : tensor
            Quantized tensor
        encoding_onehot : tensor
            Quantized tensor in onehot vector
        softmax_loss : tensor
            The loss between distance and quantized indices

        Returns
        -------
        loss : tensor
            Total loss from the VectorQuantizer layer
        perplexity : tensor
            Perplexity (i.e. entropy of quantized vectors)

        """
        # compute losses
        commitment_loss = torch.mean((z_quantized.detach() - z) ** 2)
        quantization_loss = torch.mean((z_quantized - z.detach()) ** 2)
        loss = quantization_loss + self.commitment_cost * commitment_loss + softmax_loss

        # perplexity
        avg_probs = torch.mean(encoding_onehot.float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, perplexity

    def forward(self, z):
        """

        Parameters
        ----------
        z : tensor
            input tensor; usually the output of an encoder

        Returns
        -------
        loss : length of 0
        quantized embeddings : same shape as z
        perplexity : length of 0
        encoding_onehot : shape of (Batch, Code, Width, Height)
        encoding_indices : shape of (Batch, Width, Height)
        index_histogram : shape of (Batch, Code index)

        """
        distances = self._calc_dist(z)
        # Use softmax as argmax so that loss can propagate through "histogram" process.
        softmax_histogram = torch.sum(nn.Softmax(-1)(-distances).view((z.shape[0], -1, self.num_embeddings)), dim=1)

        # Find the closest encodings
        encoding_indices = torch.argmin(distances, dim=1)
        # Pushing dist_softmax closer to encodeing_onehot
        softmax_loss = nn.CrossEntropyLoss()(-distances, encoding_indices)

        # Count histogram
        index_histogram = torch.stack(
            list(
                map(
                    lambda x: torch.histc(x, bins=self.num_embeddings, min=0, max=self.num_embeddings - 1),
                    encoding_indices.view((z.shape[0], -1)).float(),
                )
            )
        )

        # Create one-hot vectors
        encoding_onehot = nn.functional.one_hot(encoding_indices, self.num_embeddings)

        # get quantized latent vectors
        z_quantized = torch.matmul(encoding_onehot.float(), self.codebook.weight)

        # reshape back to match original input shape
        z_quantized = torch.movedim(
            z_quantized.view((z.shape[0],) + z.shape[2:] + (self.embedding_dim,)), -1, 1
        ).contiguous()

        # compute metrics
        loss, perplexity = self._calc_metrics(z, z_quantized, encoding_onehot, softmax_loss)
        # reshape back to match original input shape
        encoding_onehot = torch.movedim(
            encoding_onehot.view((z.shape[0],) + z.shape[2:] + (self.num_embeddings,)), -1, 1
        )

        # copy the gradient from inputs to quantized z.
        z_quantized = z + (z_quantized - z).detach()

        return (
            loss,
            z_quantized,
            perplexity,
            encoding_onehot,
            encoding_indices.view((-1,) + z.shape[2:]),
            index_histogram,
            softmax_histogram,
        )

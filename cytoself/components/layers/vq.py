from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn


def split_channel(z: Tensor, channel_split: int, embedding_dim: int) -> Tensor:
    """
    Split channel

    Parameters
    ----------
    z : torch.Tensor
        Pytorch tensor
    channel_split : int
        Number to be split
    embedding_dim : int
        Target embedding dimension

    Returns
    -------
    torch.Tensor

    """
    if z.shape[1] / channel_split == embedding_dim:
        z_t = torch.movedim(z, 1, -1)
        return torch.movedim(
            torch.reshape(z_t, z_t.shape[:-2] + (z_t.shape[-2] * channel_split, z_t.shape[-1] // channel_split)), -1, 1
        )
    else:
        raise ValueError(
            f'The channel dim in z (i.e. {z.shape[1]}) must be a multiple of channel_split (i.e. {channel_split} '
            f'of embedding_dim (i.e. {embedding_dim}).'
        )


def unsplit_channel(z: Tensor, channel_split: int) -> Tensor:
    """
    Undo channel splitting

    Parameters
    ----------
    z : torch.Tensor
        Pytorch tensor
    channel_split : int
        Number to be split

    Returns
    -------
    torch.Tensor

    """
    z_t = torch.movedim(z, 1, -1)
    return torch.movedim(
        torch.reshape(z_t, z_t.shape[:-2] + (z_t.shape[-2] // channel_split, z_t.shape[-1] * channel_split)), -1, 1
    )


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer
    Ref. https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        channel_split: int = 1,
        commitment_cost: float = 0.25,
        softmaxloss_cost: float = 0,
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
        channel_split : int
            Number of split in the channel dimension of input tensor
        commitment_cost : float
            Commitment cost
        softmaxloss_cost : float
            Coefficient for softmax loss
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
        self.channel_split = channel_split
        self.commitment_cost = commitment_cost
        self.softmaxloss_cost = softmaxloss_cost
        self.padding_idx = padding_idx

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim, self.padding_idx)
        if initializer == 'uniform':
            limit = np.sqrt(3.0 / self.num_embeddings)
            self.codebook.weight.data.uniform_(-limit, limit)

    def _calc_dist(self, z: Tensor) -> Tensor:
        """
        Computes distance between inputs and codebook.

        Parameters
        ----------
        z : torch.Tensor
            Usually the output of encoder

        Returns
        -------
        torch.Tensor

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

    def _calc_metrics(
        self, z: Tensor, z_quantized: Tensor, encoding_onehot: Tensor, softmax_loss: float = 0
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute metrics and losses

        Parameters
        ----------
        z : torch.Tensor
            Usually the output of encoder
        z_quantized : torch.Tensor
            Quantized tensor
        encoding_onehot : torch.Tensor
            Quantized tensor in onehot vector
        softmax_loss : torch.Tensor
            The loss between distance and quantized indices

        Returns
        -------
        loss : torch.Tensor
            Total loss from the VectorQuantizer layer
        perplexity : torch.Tensor
            Perplexity (i.e. entropy of quantized vectors)
        commitment_loss : torch.Tensor
            Commitment loss (i.e. regularizer to keep the same quantization index )
        quantization_loss : torch.Tensor
            Quantization loss (i.e. making better quantization)
        """
        # compute losses
        commitment_loss = torch.mean((z_quantized.detach() - z) ** 2)
        quantization_loss = torch.mean((z_quantized - z.detach()) ** 2)
        loss = quantization_loss + self.commitment_cost * commitment_loss + self.softmaxloss_cost * softmax_loss

        # perplexity
        avg_probs = torch.mean(encoding_onehot.float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, perplexity.detach(), commitment_loss.detach(), quantization_loss.detach()

    def forward(self, z: Tensor):
        """

        Parameters
        ----------
        z : torch.Tensor
            input tensor; usually the output of an encoder

        Returns
        -------
        A tuple of dict & tensors
        loss : a dict of losses
        quantized embeddings : same shape as z
        perplexity : length of 0
        encoding_onehot : shape of (Batch, Code * channel_split, Width, Height)
        encoding_indices : shape of (Batch, channel_split, Width, Height)
        index_histogram : shape of (Batch, Code index)
        softmax_histogram : shape of (Batch, Code index)

        """
        if self.channel_split > 1:
            z = split_channel(z, self.channel_split, self.embedding_dim)

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
        loss, perplexity, commitment_loss, quantization_loss = self._calc_metrics(
            z, z_quantized, encoding_onehot, softmax_loss
        )
        # reshape back to match original input shape
        encoding_onehot = torch.movedim(
            encoding_onehot.view((z.shape[0],) + z.shape[2:] + (self.num_embeddings,)), -1, 1
        )

        # copy the gradient from inputs to quantized z.
        z_quantized = z + (z_quantized - z).detach()

        return (
            {
                'loss': loss,
                'commitment_loss': commitment_loss,
                'quantization_loss': quantization_loss,
                'softmax_loss': softmax_loss.detach(),
            },
            unsplit_channel(z_quantized, self.channel_split),
            perplexity,
            unsplit_channel(encoding_onehot, self.channel_split),
            unsplit_channel(encoding_indices.view((-1, 1) + z.shape[2:]), self.channel_split),
            index_histogram,
            softmax_histogram,
        )

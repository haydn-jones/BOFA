from math import log

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lolbo.utils.mol_utils.selfies_vae.data import SELFIESDataset

BATCH_SIZE = 256
ENCODER_LR = 1e-3
DECODER_LR = 1e-3
ENCODER_WARMUP_STEPS = 100
DECODER_WARMUP_STEPS = 100
AGGRESSIVE_STEPS = 5


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    dim: int = -1,
) -> Tensor:
    """
    Mostly from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
    """
    randoms = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + randoms) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class InfoTransformerVAE(pl.LightningModule):
    def __init__(
        self,
        dataset: SELFIESDataset,
        bottleneck_size: int = 2,
        d_model: int = 128,
        is_autoencoder: bool = False,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 512,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
        decoder_nhead: int = 8,
        decoder_dim_feedforward: int = 256,
        decoder_dropout: float = 0.1,
        decoder_num_layers: int = 6,
    ):
        super().__init__()

        assert (
            bottleneck_size is not None
        ), "Dont set bottleneck_size to None. Unbounded sequences dont support this yet"

        self.max_string_length = 256

        self.dataset = dataset
        self.vocab_size = len(self.dataset.vocab)

        self.bottleneck_size = bottleneck_size
        self.d_model = d_model
        self.is_autoencoder = is_autoencoder

        # TODO
        self.kl_factor = kl_factor

        self.min_posterior_std = min_posterior_std
        encoder_embedding_dim = 2 * d_model

        self.encoder_token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=encoder_embedding_dim)
        self.encoder_position_encoding = PositionalEncoding(
            encoder_embedding_dim, dropout=encoder_dropout, max_len=5_000
        )
        self.decoder_token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=d_model)
        self.decoder_position_encoding = PositionalEncoding(d_model, dropout=decoder_dropout, max_len=5_000)
        self.decoder_token_unembedding = nn.Parameter(torch.randn(d_model, self.vocab_size))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_embedding_dim,
                nhead=encoder_nhead,
                dim_feedforward=encoder_dim_feedforward,
                dropout=encoder_dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=encoder_num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=decoder_nhead,
                dim_feedforward=decoder_dim_feedforward,
                dropout=decoder_dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=decoder_num_layers,
        )

    def sample_prior(self, n):
        if self.bottleneck_size is None:
            # TODO: idk what to do there lol, seq len doesn't exist anymore
            sequence_length = self.sequence_length
        else:
            sequence_length = self.bottleneck_size

        return torch.randn(n, sequence_length, self.d_model).to(self.device)

    def sample_posterior(self, mu, sigma, n=None):
        if n is not None:
            mu = mu.unsqueeze(0).expand(n, -1, -1, -1)

        return mu + torch.randn_like(mu) * sigma

    def generate_pad_mask(self, tokens):
        """Generate mask that tells encoder to ignore all but first stop token"""
        mask = tokens == 1
        inds = mask.float().argmax(dim=-1)  # Returns first index along axis when multiple present
        mask[torch.arange(0, tokens.shape[0]), inds] = False
        return mask

    def encode(self, tokens, as_probs=False):
        if as_probs:
            embed = tokens @ self.encoder_token_embedding.weight
        else:
            embed = self.encoder_token_embedding(tokens)

        embed = self.encoder_position_encoding(embed)

        pad_mask = self.generate_pad_mask(tokens)
        encoding = self.encoder(embed, src_key_padding_mask=pad_mask)
        mu = encoding[..., : self.d_model]
        sigma = F.softplus(encoding[..., self.d_model :]) + self.min_posterior_std

        if self.bottleneck_size is not None:
            mu = mu[:, : self.bottleneck_size, :]
            sigma = sigma[:, : self.bottleneck_size, :]

        return mu, sigma

    def decode(self, z, tokens):
        embed = self.decoder_token_embedding(tokens[:, :-1])
        embed = torch.cat(
            [
                # Zero is the start token
                torch.zeros(embed.shape[0], 1, embed.shape[-1], device=self.device),
                embed,
            ],
            dim=1,
        )
        embed = self.decoder_position_encoding(embed)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(embed.shape[1]).to(self.device)
        decoding = self.decoder(tgt=embed, memory=z, tgt_mask=tgt_mask)
        logits = decoding @ self.decoder_token_unembedding

        return logits

    @torch.no_grad()
    def sample(self, z: Tensor) -> Tensor:
        model_state = self.training
        self.eval()
        n = z.shape[0]

        tokens = torch.zeros(n, 1, device=self.device).long()  # Start token is 0, stop token is 1
        while True:  # Loop until every molecule hits a stop token
            tgt = self.decoder_token_embedding(tokens)
            tgt = self.decoder_position_encoding(tgt)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[-1], device=self.device)

            decoding = self.decoder(tgt=tgt, memory=z, tgt_mask=tgt_mask)[:, -1:]
            logits = decoding @ self.decoder_token_unembedding
            sample = gumbel_softmax(logits, dim=-1, hard=True)

            tokens = torch.cat([tokens, sample.argmax(dim=-1)], dim=-1)

            if torch.all(torch.any(tokens == 1, dim=-1)).item() or tokens.shape[-1] > self.max_string_length:
                break

        self.train(model_state)

        return tokens

    def forward(self, tokens):
        mu, sigma = self.encode(tokens)

        z = self.sample_posterior(mu, sigma)

        logits = self.decode(z, tokens)

        recon_loss = F.cross_entropy(logits.permute(0, 2, 1), tokens, reduction="none").mean()  # .sum(1).mean(0)

        sigma2 = sigma.pow(2)
        kldiv = 0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1).mean()

        primary_loss = recon_loss
        if self.kl_factor != 0:
            primary_loss = primary_loss + self.kl_factor * kldiv
        loss = primary_loss

        return dict(
            loss=loss,
            z=z,
            recon_loss=recon_loss,
            kldiv=kldiv,
            recon_token_acc=(logits.argmax(dim=-1) == tokens).float().mean(),
            recon_string_acc=(logits.argmax(dim=-1) == tokens).all(dim=1).float().mean(dim=0),
            sigma_mean=sigma.mean(),
        )


class VAEModule(pl.LightningModule):
    def __init__(
        self,
        dataset: SELFIESDataset,
        bottleneck_size: int = 2,
        d_model: int = 128,
        is_autoencoder: bool = False,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 512,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
        decoder_nhead: int = 8,
        decoder_dim_feedforward: int = 256,
        decoder_dropout: float = 0.1,
        decoder_num_layers: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="dataset")
        self.model = InfoTransformerVAE(dataset=dataset, **self.hparams)

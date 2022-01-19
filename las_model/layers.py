import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    r"""Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.
    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, values):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H
        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        input_lengths = values.size(1)
        # (N, To, H) * (N, H, Ti) -> (N, To, Ti)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        attention_distribution = F.softmax(
            attention_scores.view(-1, input_lengths), dim=1).view(batch_size, -1, input_lengths)
        # (N, To, Ti) * (N, Ti, H) -> (N, To, H)
        attention_output = torch.bmm(attention_distribution, values)

        return attention_output, attention_distribution


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, query, value):
        score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        self.query_projection = nn.Linear(hidden_dim, self.dim * num_heads)
        self.value_projection = nn.Linear(hidden_dim, self.dim * num_heads)
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)

    def forward(self, query, value, prev_attn=None):
        batch_size = value.size(0)
        residual = query

        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.dim)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.dim)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)

        context, attn = self.scaled_dot(query, value)
        context = context.view(self.num_heads, batch_size, -1, self.dim)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.dim)
        combined = torch.cat([context, residual], dim=2)

        output = torch.tanh(self.out_projection(combined.view(-1, self.hidden_dim << 1))).view(batch_size, -1, self.hidden_dim)
        return output, context


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.

    Args:
        decoder_dim (int): dimension of model
        attn_dim (int): dimension of attention
        smoothing (bool): flag indication whether to use smoothing or not.

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """
    def __init__(self, decoder_dim: int = 1024, attn_dim: int = 1024, smoothing: bool = False) -> None:
        super(LocationAwareAttention, self).__init__()
        self.decoder_dim = decoder_dim
        self.attn_dim = attn_dim
        self.location_conv = nn.Conv1d(in_channels=1, out_channels=attn_dim, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.value_proj = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))
        self.fc = nn.Linear(attn_dim, 1, bias=True)
        self.smoothing = smoothing

    def forward(self, query, value, last_alignment_energy):
        batch_size, hidden_dim, seq_length = query.size(0), query.size(2), value.size(1)

        if last_alignment_energy is None:
            last_alignment_energy = value.new_zeros(batch_size, seq_length)

        last_alignment_energy = self.location_conv(last_alignment_energy.unsqueeze(dim=1))
        last_alignment_energy = last_alignment_energy.transpose(1, 2)

        alignmment_energy = self.fc(torch.tanh(
                self.query_proj(query)
                + self.value_proj(value)
                + last_alignment_energy
                + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            alignmment_energy = torch.sigmoid(alignmment_energy)
            alignmment_energy = torch.div(alignmment_energy, alignmment_energy.sum(dim=-1).unsqueeze(dim=-1))

        else:
            alignmment_energy = F.softmax(alignmment_energy, dim=-1)

        context = torch.bmm(alignmment_energy.unsqueeze(dim=1), value)

        return context, alignmment_energy

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, ignore_index, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, logit, target):
        with torch.no_grad():
            label_smoothed = torch.zeros_like(logit).cuda()
            label_smoothed.fill_(self.smoothing / (self.vocab_size - 1))
            #print(label_smoothed, target.data.unsqueeze(1))
            label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
            label_smoothed[target == self.ignore_index, :] = 0

            # print(label_smoothed)
            # print(label_smoothed.size())
            # print(logit)
            # print(logit.size())
            # print(-label_smoothed * logit)
            # print(torch.sum(-label_smoothed * logit))

        return torch.sum(-label_smoothed * logit)
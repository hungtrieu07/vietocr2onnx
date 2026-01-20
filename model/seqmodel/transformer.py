"""
Transformer-based sequence model for VietOCR.
This implementation supports ONNX export for inference.
"""

import math
import torch
from torch import nn


class LanguageTransformer(nn.Module):
    """
    Transformer-based language model for OCR.
    
    This model uses PyTorch's built-in Transformer and supports:
    - Full forward pass for training
    - Separate encoder/decoder forward for inference
    - ONNX export compatibility
    """
    
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        pos_dropout,
        trans_dropout,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Target embedding
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            trans_dropout,
        )

        # Output projection
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Full forward pass for training.
        
        Shape:
            - src: (W, N, C) - sequence length, batch, channels
            - tgt: (T, N) - target length, batch
            - src_key_padding_mask: (N, S)
            - tgt_key_padding_mask: (N, T)
            - memory_key_padding_mask: (N, S)
            - output: (N, T, vocab_size)
        """
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(src.device)

        src = self.pos_enc(src * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask.float() if tgt_key_padding_mask is not None else None,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        output = output.transpose(0, 1)  # (T, N, E) -> (N, T, E)
        return self.fc(output)

    def gen_nopeek_mask(self, length):
        """Generate causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(length, length), diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float("-inf"))
        return mask

    def forward_encoder(self, src):
        """
        Encoder forward pass.
        
        Args:
            src: Source features from CNN (seq_len, batch, d_model)
            
        Returns:
            memory: Encoder output (seq_len, batch, d_model)
        """
        src = self.pos_enc(src * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src)
        return memory

    def forward_decoder(self, tgt, memory):
        """
        Decoder forward pass for autoregressive inference.
        
        Args:
            tgt: Target tokens so far (seq_len, batch)
            memory: Encoder output from forward_encoder
            
        Returns:
            output: Logits (batch, seq_len, vocab_size)
            memory: Same memory (for compatibility with inference loop)
        """
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)  # (T, N, E) -> (N, T, E)

        return self.fc(output), memory

    def expand_memory(self, memory, beam_size):
        """Expand memory for beam search."""
        memory = memory.repeat(1, beam_size, 1)
        return memory

    def get_memory(self, memory, i):
        """Get memory for a specific beam."""
        memory = memory[:, [i], :]
        return memory


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor (seq_len, batch, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# Wrapper classes for ONNX export
class TransformerEncoderWrapper(nn.Module):
    """Wrapper for exporting only the encoder part to ONNX."""
    
    def __init__(self, transformer_model):
        super().__init__()
        self.d_model = transformer_model.d_model
        self.pos_enc = transformer_model.pos_enc
        self.encoder = transformer_model.transformer.encoder
    
    def forward(self, src):
        """
        Args:
            src: Source features (seq_len, batch, d_model)
        Returns:
            memory: Encoder output (seq_len, batch, d_model)
        """
        src = self.pos_enc(src * math.sqrt(self.d_model))
        return self.encoder(src)


class TransformerDecoderWrapper(nn.Module):
    """Wrapper for exporting only the decoder part to ONNX."""
    
    def __init__(self, transformer_model):
        super().__init__()
        self.d_model = transformer_model.d_model
        self.embed_tgt = transformer_model.embed_tgt
        self.pos_enc = transformer_model.pos_enc
        self.decoder = transformer_model.transformer.decoder
        self.fc = transformer_model.fc
    
    def gen_nopeek_mask(self, length, device):
        """Generate causal mask."""
        mask = torch.triu(torch.ones(length, length, device=device), diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float("-inf"))
        return mask
    
    def forward(self, tgt, memory):
        """
        Args:
            tgt: Target tokens (seq_len, batch)
            memory: Encoder output (src_len, batch, d_model)
        Returns:
            output: Logits (batch, seq_len, vocab_size)
        """
        tgt_len = tgt.shape[0]
        tgt_mask = self.gen_nopeek_mask(tgt_len, tgt.device)
        
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)  # (T, N, E) -> (N, T, E)
        
        return self.fc(output)

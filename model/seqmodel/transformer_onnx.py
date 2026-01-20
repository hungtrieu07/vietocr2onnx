"""
ONNX-Compatible Transformer for VietOCR.

This module provides a custom Transformer implementation that avoids the 
dynamic shape issues in PyTorch's nn.Transformer when exporting to ONNX.

The key difference is that this implementation:
1. Uses einsum/matmul instead of view/reshape for attention computation
2. Pre-allocates causal masks up to max sequence length
3. Avoids operations that bake in sequence length during tracing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ONNXMultiheadAttention(nn.Module):
    """
    ONNX-compatible Multi-head Attention.
    
    Uses matrix operations that work with dynamic shapes in ONNX.
    """
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: (seq_len, batch, d_model)
            key: (seq_len, batch, d_model)
            value: (seq_len, batch, d_model)
            attn_mask: (seq_len, seq_len) or None - additive mask
            key_padding_mask: (batch, seq_len) or None
        Returns:
            output: (seq_len, batch, d_model)
        """
        seq_len, batch_size, _ = query.shape
        
        # Project to Q, K, V
        q = self.q_proj(query)  # (seq, batch, d_model)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to (batch, nhead, seq, head_dim) using transpose instead of view
        # First: (seq, batch, d_model) -> (batch, seq, d_model)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # Then: (batch, seq, d_model) -> (batch, seq, nhead, head_dim) -> (batch, nhead, seq, head_dim)
        q = q.reshape(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: (batch, nhead, seq_q, head_dim) x (batch, nhead, head_dim, seq_k)
        # -> (batch, nhead, seq_q, seq_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask (additive mask, -inf for masked positions)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        # Apply key padding mask
        if key_padding_mask is not None:
            # (batch, seq_k) -> (batch, 1, 1, seq_k)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: (batch, nhead, seq_q, seq_k) x (batch, nhead, seq_k, head_dim)
        # -> (batch, nhead, seq_q, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch, nhead, seq, head_dim) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        
        # (batch, seq, d_model) -> (seq, batch, d_model)
        attn_output = attn_output.transpose(0, 1)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class ONNXTransformerEncoderLayer(nn.Module):
    """ONNX-compatible Transformer Encoder Layer."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = ONNXMultiheadAttention(d_model, nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, 
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class ONNXTransformerDecoderLayer(nn.Module):
    """ONNX-compatible Transformer Decoder Layer."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = ONNXMultiheadAttention(d_model, nhead, dropout)
        self.cross_attn = ONNXMultiheadAttention(d_model, nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class ONNXTransformerEncoder(nn.Module):
    """ONNX-compatible Transformer Encoder."""
    
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ONNXTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # Final layer norm (matches PyTorch TransformerEncoder)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        # Apply final norm
        output = self.norm(output)
        return output


class ONNXTransformerDecoder(nn.Module):
    """ONNX-compatible Transformer Decoder."""
    
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ONNXTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # Final layer norm (matches PyTorch TransformerDecoder)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        # Apply final norm
        output = self.norm(output)
        return output


class ONNXPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ONNXLanguageTransformer(nn.Module):
    """
    ONNX-compatible Language Transformer for VietOCR.
    
    This is a drop-in replacement for the original LanguageTransformer
    that works with ONNX dynamic shapes.
    """
    
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, max_seq_length, 
                 pos_dropout, trans_dropout):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_enc = ONNXPositionalEncoding(d_model, pos_dropout, max_seq_length)
        
        # Transformer encoder and decoder
        self.encoder = ONNXTransformerEncoder(
            d_model, nhead, num_encoder_layers, dim_feedforward, trans_dropout
        )
        self.decoder = ONNXTransformerDecoder(
            d_model, nhead, num_decoder_layers, dim_feedforward, trans_dropout
        )
        
        # Output projection
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Pre-computed causal mask
        causal_mask = torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1)
        causal_mask = causal_mask.float().masked_fill(causal_mask == 1, float('-inf'))
        self.register_buffer('causal_mask', causal_mask)
    
    def gen_nopeek_mask(self, length):
        """Generate causal mask for decoder."""
        return self.causal_mask[:length, :length]
    
    def forward_encoder(self, src):
        """Encode source features."""
        src = self.pos_enc(src * math.sqrt(self.d_model))
        return self.encoder(src)
    
    def forward_decoder(self, tgt, memory):
        """Decode with cross-attention to encoder output."""
        tgt_len = tgt.shape[0]
        tgt_mask = self.gen_nopeek_mask(tgt_len)
        
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)  # (seq, batch, d) -> (batch, seq, d)
        
        return self.fc(output), memory
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """Full forward pass."""
        tgt_len = tgt.shape[0]
        tgt_mask = self.gen_nopeek_mask(tgt_len)
        
        src = self.pos_enc(src * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask)
        
        output = output.transpose(0, 1)
        return self.fc(output)
    
    # Methods for beam search compatibility
    def expand_memory(self, memory, beam_size):
        return memory.repeat(1, beam_size, 1)
    
    def get_memory(self, memory, i):
        return memory[:, [i], :]


def load_weights_from_pytorch_transformer(onnx_model, pytorch_model):
    """
    Copy weights from a PyTorch nn.Transformer-based model to ONNX-compatible model.
    
    Args:
        onnx_model: ONNXLanguageTransformer instance
        pytorch_model: Original LanguageTransformer with nn.Transformer
    """
    # Copy embedding
    onnx_model.embed_tgt.load_state_dict(pytorch_model.embed_tgt.state_dict())
    
    # Copy positional encoding (if using buffer)
    if hasattr(pytorch_model, 'pos_enc') and hasattr(pytorch_model.pos_enc, 'pe'):
        # Resize if needed
        max_len = min(onnx_model.pos_enc.pe.shape[0], pytorch_model.pos_enc.pe.shape[0])
        onnx_model.pos_enc.pe[:max_len] = pytorch_model.pos_enc.pe[:max_len]
    
    # Copy output projection
    onnx_model.fc.load_state_dict(pytorch_model.fc.state_dict())
    
    # Copy encoder layers
    pt_encoder = pytorch_model.transformer.encoder
    for i, (onnx_layer, pt_layer) in enumerate(zip(onnx_model.encoder.layers, pt_encoder.layers)):
        # Self-attention projections
        # PyTorch uses in_proj_weight/bias that combines Q, K, V
        if hasattr(pt_layer.self_attn, 'in_proj_weight'):
            d_model = onnx_model.d_model
            in_proj = pt_layer.self_attn.in_proj_weight
            in_proj_bias = pt_layer.self_attn.in_proj_bias
            
            onnx_layer.self_attn.q_proj.weight.data = in_proj[:d_model]
            onnx_layer.self_attn.k_proj.weight.data = in_proj[d_model:2*d_model]
            onnx_layer.self_attn.v_proj.weight.data = in_proj[2*d_model:]
            
            if in_proj_bias is not None:
                onnx_layer.self_attn.q_proj.bias.data = in_proj_bias[:d_model]
                onnx_layer.self_attn.k_proj.bias.data = in_proj_bias[d_model:2*d_model]
                onnx_layer.self_attn.v_proj.bias.data = in_proj_bias[2*d_model:]
        
        onnx_layer.self_attn.out_proj.load_state_dict(pt_layer.self_attn.out_proj.state_dict())
        
        # FFN
        onnx_layer.linear1.load_state_dict(pt_layer.linear1.state_dict())
        onnx_layer.linear2.load_state_dict(pt_layer.linear2.state_dict())
        onnx_layer.norm1.load_state_dict(pt_layer.norm1.state_dict())
        onnx_layer.norm2.load_state_dict(pt_layer.norm2.state_dict())
    
    # Copy decoder layers
    pt_decoder = pytorch_model.transformer.decoder
    for i, (onnx_layer, pt_layer) in enumerate(zip(onnx_model.decoder.layers, pt_decoder.layers)):
        d_model = onnx_model.d_model
        
        # Self-attention
        if hasattr(pt_layer.self_attn, 'in_proj_weight'):
            in_proj = pt_layer.self_attn.in_proj_weight
            in_proj_bias = pt_layer.self_attn.in_proj_bias
            
            onnx_layer.self_attn.q_proj.weight.data = in_proj[:d_model]
            onnx_layer.self_attn.k_proj.weight.data = in_proj[d_model:2*d_model]
            onnx_layer.self_attn.v_proj.weight.data = in_proj[2*d_model:]
            
            if in_proj_bias is not None:
                onnx_layer.self_attn.q_proj.bias.data = in_proj_bias[:d_model]
                onnx_layer.self_attn.k_proj.bias.data = in_proj_bias[d_model:2*d_model]
                onnx_layer.self_attn.v_proj.bias.data = in_proj_bias[2*d_model:]
        
        onnx_layer.self_attn.out_proj.load_state_dict(pt_layer.self_attn.out_proj.state_dict())
        
        # Cross-attention
        if hasattr(pt_layer.multihead_attn, 'in_proj_weight'):
            in_proj = pt_layer.multihead_attn.in_proj_weight
            in_proj_bias = pt_layer.multihead_attn.in_proj_bias
            
            onnx_layer.cross_attn.q_proj.weight.data = in_proj[:d_model]
            onnx_layer.cross_attn.k_proj.weight.data = in_proj[d_model:2*d_model]
            onnx_layer.cross_attn.v_proj.weight.data = in_proj[2*d_model:]
            
            if in_proj_bias is not None:
                onnx_layer.cross_attn.q_proj.bias.data = in_proj_bias[:d_model]
                onnx_layer.cross_attn.k_proj.bias.data = in_proj_bias[d_model:2*d_model]
                onnx_layer.cross_attn.v_proj.bias.data = in_proj_bias[2*d_model:]
        
        onnx_layer.cross_attn.out_proj.load_state_dict(pt_layer.multihead_attn.out_proj.state_dict())
        
        # FFN
        onnx_layer.linear1.load_state_dict(pt_layer.linear1.state_dict())
        onnx_layer.linear2.load_state_dict(pt_layer.linear2.state_dict())
        onnx_layer.norm1.load_state_dict(pt_layer.norm1.state_dict())
        onnx_layer.norm2.load_state_dict(pt_layer.norm2.state_dict())
        onnx_layer.norm3.load_state_dict(pt_layer.norm3.state_dict())
    
    # Copy final norm layers for encoder and decoder
    if hasattr(pt_encoder, 'norm') and pt_encoder.norm is not None:
        onnx_model.encoder.norm.load_state_dict(pt_encoder.norm.state_dict())
    
    if hasattr(pt_decoder, 'norm') and pt_decoder.norm is not None:
        onnx_model.decoder.norm.load_state_dict(pt_decoder.norm.state_dict())
    
    print("Successfully loaded weights into ONNX-compatible model")

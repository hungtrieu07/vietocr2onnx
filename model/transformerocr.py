"""
VietOCR main model that combines CNN backbone with sequence modeling.
Supports both Seq2Seq (GRU-based) and Transformer architectures.
"""

from model.backbone.cnn import CNN
from model.seqmodel.seq2seq import Seq2Seq
from model.seqmodel.transformer import LanguageTransformer
from torch import nn


class VietOCR(nn.Module):
    """
    VietOCR model combining CNN feature extractor with sequence model.
    
    Args:
        vocab_size: Size of the vocabulary
        backbone: CNN backbone type (e.g., 'vgg19_bn')
        cnn_args: Arguments for CNN backbone
        transformer_args: Arguments for sequence model
        seq_modeling: Type of sequence model ('seq2seq' or 'transformer')
    """
    
    def __init__(self, vocab_size,
                 backbone,
                 cnn_args, 
                 transformer_args, 
                 seq_modeling='transformer'):
        
        super(VietOCR, self).__init__()
        
        self.cnn = CNN(backbone, **cnn_args)
        self.seq_modeling = seq_modeling
        
        if seq_modeling == 'transformer':
            self.transformer = LanguageTransformer(vocab_size, **transformer_args)
        else:  # seq2seq
            self.transformer = Seq2Seq(vocab_size, **transformer_args)

    def forward(self, img, tgt_input, tgt_key_padding_mask=None):
        """
        Forward pass for training.
        
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T) - only used for transformer
            - output: (N, T, vocab_size)
        """
        src = self.cnn(img)
        
        if self.seq_modeling == 'transformer':
            outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            outputs = self.transformer(src, tgt_input)

        return outputs
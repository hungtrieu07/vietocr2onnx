"""
VietOCR Transformer to ONNX Converter - Separate Encoder/Decoder.

This version exports TWO ONNX models:
1. Encoder (CNN + Transformer Encoder) - run once per image
2. Decoder (Transformer Decoder) - run for each token generation step

This allows caching the encoder output (memory) during autoregressive decoding.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from tool.config import Cfg
from tool.translate import build_model, process_input
from model.vocab import Vocab
from model.seqmodel.transformer_onnx import ONNXLanguageTransformer, load_weights_from_pytorch_transformer


class EncoderONNX(nn.Module):
    """CNN + Transformer Encoder for ONNX export."""
    
    def __init__(self, cnn, transformer):
        super().__init__()
        self.cnn = cnn
        self.transformer = transformer
    
    def forward(self, img):
        """
        Args:
            img: Input image (batch, 3, H, W)
        Returns:
            memory: Encoder output (seq_len, batch, d_model)
        """
        src = self.cnn(img)
        memory = self.transformer.forward_encoder(src)
        return memory


class DecoderONNX(nn.Module):
    """Transformer Decoder for ONNX export."""
    
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
    
    def forward(self, tgt, memory):
        """
        Args:
            tgt: Target tokens (seq_len, batch)
            memory: Encoder output (src_len, batch, d_model)
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
        """
        output, _ = self.transformer.forward_decoder(tgt, memory)
        return output


def create_onnx_model(config, vocab_size):
    """Create ONNX-compatible Transformer model from config."""
    transformer_config = config['transformer']
    
    # Use dropout=0 for ONNX export since inference doesn't use dropout
    # This ensures outputs match PyTorch model in eval mode
    onnx_transformer = ONNXLanguageTransformer(
        vocab_size=vocab_size,
        d_model=transformer_config['d_model'],
        nhead=transformer_config['nhead'],
        num_encoder_layers=transformer_config['num_encoder_layers'],
        num_decoder_layers=transformer_config['num_decoder_layers'],
        dim_feedforward=transformer_config['dim_feedforward'],
        max_seq_length=transformer_config.get('max_seq_length', 1024),
        pos_dropout=0.0,  # Disable dropout for inference
        trans_dropout=0.0  # Disable dropout for inference
    )
    
    return onnx_transformer


def convert_encoder(encoder, save_path, sample_img):
    """Export encoder to ONNX."""
    print("Exporting Encoder (CNN + Transformer Encoder)...")
    
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(sample_img)
        print(f"  Input shape: {sample_img.shape}")
        print(f"  Output shape: {output.shape}")
        
        torch.onnx.export(
            encoder,
            sample_img,
            save_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['memory'],
            dynamic_axes={
                'image': {0: 'batch', 3: 'width'},
                'memory': {0: 'seq_len', 1: 'batch'}
            }
        )
    
    print(f"  ✓ Saved to {save_path}")
    return output


def convert_decoder(decoder, save_path, sample_tgt, sample_memory):
    """Export decoder to ONNX."""
    print("Exporting Decoder...")
    
    decoder.eval()
    
    with torch.no_grad():
        output = decoder(sample_tgt, sample_memory)
        print(f"  Target shape: {sample_tgt.shape}")
        print(f"  Memory shape: {sample_memory.shape}")
        print(f"  Output shape: {output.shape}")
        
        torch.onnx.export(
            decoder,
            (sample_tgt, sample_memory),
            save_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['tgt', 'memory'],
            output_names=['logits'],
            dynamic_axes={
                'tgt': {0: 'tgt_len', 1: 'batch'},
                'memory': {0: 'src_len', 1: 'batch'},
                'logits': {0: 'batch', 1: 'tgt_len'}
            }
        )
    
    print(f"  ✓ Saved to {save_path}")


def verify_models(encoder_path, decoder_path, sample_img, sample_tgt, encoder, decoder):
    """Verify ONNX models match PyTorch outputs."""
    import onnx
    import onnxruntime
    
    print("\nVerifying ONNX models...")
    
    # Check validity
    onnx.checker.check_model(onnx.load(encoder_path))
    onnx.checker.check_model(onnx.load(decoder_path))
    print("  ✓ Both models are structurally valid")
    
    # Load sessions
    enc_session = onnxruntime.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
    dec_session = onnxruntime.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])
    
    # Compare encoder
    with torch.no_grad():
        pt_memory = encoder(sample_img).numpy()
    
    onnx_memory = enc_session.run(None, {'image': sample_img.numpy()})[0]
    enc_diff = np.abs(pt_memory - onnx_memory).max()
    print(f"  Encoder max diff: {enc_diff:.6f}")
    
    # Compare decoder
    with torch.no_grad():
        pt_logits = decoder(sample_tgt, torch.from_numpy(onnx_memory)).numpy()
    
    onnx_logits = dec_session.run(None, {
        'tgt': sample_tgt.numpy(),
        'memory': onnx_memory
    })[0]
    dec_diff = np.abs(pt_logits - onnx_logits).max()
    print(f"  Decoder max diff: {dec_diff:.6f}")
    
    if enc_diff < 1e-4 and dec_diff < 1e-4:
        print("  ✓ All outputs match!")
    else:
        print("  ⚠ Some differences detected, but should be acceptable for inference")


def main():
    parser = argparse.ArgumentParser(description='Convert VietOCR Transformer to ONNX (Separate Encoder/Decoder)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--output_dir', type=str, default='./converted_weights_transformer/',
                       help='Directory to save ONNX models')
    parser.add_argument('--verify', action='store_true', help='Verify ONNX model outputs')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    print(f"Loading config from {args.config}...")
    config = Cfg.load_config_from_file(args.config)
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'
    
    # Build PyTorch model
    print("Building PyTorch model...")
    pytorch_model, vocab = build_model(config)
    
    print(f"Loading weights from {args.weights}...")
    pytorch_model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    pytorch_model.eval()
    
    # Create ONNX-compatible transformer
    print("\nCreating ONNX-compatible model...")
    onnx_transformer = create_onnx_model(config, len(vocab))
    
    # Transfer weights
    print("Transferring weights...")
    load_weights_from_pytorch_transformer(onnx_transformer, pytorch_model.transformer)
    onnx_transformer.eval()
    
    # Create encoder and decoder wrappers
    encoder = EncoderONNX(pytorch_model.cnn, onnx_transformer)
    decoder = DecoderONNX(onnx_transformer)
    encoder.eval()
    decoder.eval()
    
    # Sample inputs
    print("\nPreparing sample inputs...")
    max_width = config['dataset']['image_max_width']
    sample_img = torch.rand(1, 3, config['dataset']['image_height'], max_width)
    sample_tgt = torch.LongTensor([[vocab.go]])
    
    print(f"  Image: {sample_img.shape}")
    print(f"  Target: {sample_tgt.shape}")
    
    # Export encoder
    encoder_path = os.path.join(args.output_dir, 'encoder.onnx')
    sample_memory = convert_encoder(encoder, encoder_path, sample_img)
    
    # Export decoder
    decoder_path = os.path.join(args.output_dir, 'decoder.onnx')
    convert_decoder(decoder, decoder_path, sample_tgt, sample_memory)
    
    # Verify
    if args.verify:
        verify_models(encoder_path, decoder_path, sample_img, sample_tgt, encoder, decoder)
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - {encoder_path}")
    print(f"  - {decoder_path}")
    print(f"\nModel info:")
    print(f"  - Vocab size: {len(vocab)}")
    print(f"  - SOS token: {vocab.go}")
    print(f"  - EOS token: {vocab.eos}")
    print(f"\nUsage:")
    print(f"  python inference_onnx_transformer_split.py --image <image> --config {args.config}")


if __name__ == '__main__':
    main()

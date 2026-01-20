"""
ONNX Inference Script for VietOCR Transformer (Split Encoder/Decoder).

Uses separate encoder and decoder ONNX models for efficient inference:
1. Run encoder ONCE to get memory
2. Run decoder for each token in autoregressive loop

Usage:
    python inference_onnx_transformer_split.py --image <image_path> --config <config_path>
"""

import os
import argparse
import time
import math
import numpy as np
from PIL import Image
import onnxruntime

from tool.config import Cfg
from model.vocab import Vocab


def create_session(model_path):
    """Create ONNX Runtime session."""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        provider = session.get_providers()[0]
        print(f"  ✓ {os.path.basename(model_path)}: {provider}")
        return session
    except Exception as e:
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print(f"  ✓ {os.path.basename(model_path)}: CPUExecutionProvider (fallback)")
        return session


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    new_w = math.ceil(new_w / 10) * 10
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)
    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    """Process image with padding to max width."""
    img = image.convert('RGB')
    w, h = img.size
    new_w, new_h = resize(w, h, image_height, image_min_width, image_max_width)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    img_np = np.asarray(img).transpose(2, 0, 1) / 255.0
    
    # Pad to max width
    if new_w < image_max_width:
        padded = np.zeros((3, image_height, image_max_width), dtype=np.float32)
        padded[:, :, :new_w] = img_np
        img_np = padded
    
    return img_np[np.newaxis, ...].astype(np.float32)


def translate(encoder_session, decoder_session, img, max_seq_length=128, sos_token=1, eos_token=2):
    """
    Perform OCR inference with cached encoder output.
    
    1. Encode image once
    2. Decode autoregressively using cached memory
    """
    # Step 1: Encode image (run ONCE)
    memory = encoder_session.run(None, {'image': img})[0]
    
    # Step 2: Decode autoregressively
    tokens = [[sos_token]]
    
    for _ in range(max_seq_length):
        tgt = np.array(tokens, dtype=np.int64)
        
        logits = decoder_session.run(None, {
            'tgt': tgt,
            'memory': memory
        })[0]
        
        # Get next token
        next_token = np.argmax(logits[0, -1, :])
        tokens.append([int(next_token)])
        
        if next_token == eos_token:
            break
    
    return [t[0] for t in tokens]


def translate_with_confidence(encoder_session, decoder_session, img, 
                              max_seq_length=128, sos_token=1, eos_token=2):
    """Translate with confidence scores."""
    memory = encoder_session.run(None, {'image': img})[0]
    
    tokens = [[sos_token]]
    confidences = [1.0]
    
    for _ in range(max_seq_length):
        tgt = np.array(tokens, dtype=np.int64)
        logits = decoder_session.run(None, {'tgt': tgt, 'memory': memory})[0]
        
        last_logits = logits[0, -1, :]
        exp_logits = np.exp(last_logits - np.max(last_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        next_token = np.argmax(probs)
        confidence = probs[next_token]
        
        tokens.append([int(next_token)])
        confidences.append(float(confidence))
        
        if next_token == eos_token:
            break
    
    return [t[0] for t in tokens], confidences


def main():
    parser = argparse.ArgumentParser(description='VietOCR ONNX Inference (Split Encoder/Decoder)')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--encoder', type=str, default='./converted_weights_transformer/encoder.onnx')
    parser.add_argument('--decoder', type=str, default='./converted_weights_transformer/decoder.onnx')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--show_confidence', action='store_true')
    
    args = parser.parse_args()
    
    # Load config and vocab
    print(f"Loading config from {args.config}...")
    config = Cfg.load_config_from_file(args.config)
    vocab = Vocab(config['vocab'])
    
    # Load models
    print("\nLoading ONNX models...")
    encoder_session = create_session(args.encoder)
    decoder_session = create_session(args.decoder)
    
    # Process image
    print(f"\nProcessing image: {args.image}")
    img = Image.open(args.image)
    img_processed = process_image(
        img,
        config['dataset']['image_height'],
        config['dataset']['image_min_width'],
        config['dataset']['image_max_width']
    )
    print(f"  Input shape: {img_processed.shape}")
    
    # Run inference
    print("\nRunning inference...")
    start_time = time.perf_counter()
    
    if args.show_confidence:
        tokens, confidences = translate_with_confidence(
            encoder_session, decoder_session, img_processed,
            max_seq_length=args.max_seq_length,
            sos_token=vocab.go,
            eos_token=vocab.eos
        )
        avg_conf = np.mean(confidences[1:])
    else:
        tokens = translate(
            encoder_session, decoder_session, img_processed,
            max_seq_length=args.max_seq_length,
            sos_token=vocab.go,
            eos_token=vocab.eos
        )
    
    end_time = time.perf_counter()
    
    # Decode
    text = vocab.decode(tokens)
    
    print(f"\n{'='*60}")
    print(f"Result: {text}")
    print(f"{'='*60}")
    print(f"Time: {end_time - start_time:.4f}s")
    
    if args.show_confidence:
        print(f"Avg confidence: {avg_conf:.2%}")
    
    return text


if __name__ == '__main__':
    main()

# VietOCR to ONNX Converter

Chuyển đổi mô hình VietOCR sang định dạng ONNX để inference nhanh hơn.

**Tutorial gốc**: [Chuyển đổi mô hình học sâu về ONNX](https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw)

---

## Kiến trúc hỗ trợ

| Kiến trúc | Mô tả | Script chuyển đổi | Script inference |
|-----------|-------|-------------------|------------------|
| **Seq2Seq** | Encoder-decoder dựa trên GRU | `Converter.ipynb` | `inference_onnx.py` |
| **Transformer** | Encoder-decoder dựa trên self-attention | `convert_transformer_split.py` | `inference_onnx_transformer_split.py` |

---

## Hướng dẫn chuyển đổi mô hình Transformer

### Bước 1: Chuẩn bị

Đảm bảo bạn có:
- File trọng số `.pth` (ví dụ: `transformerocr.pth`)
- File cấu hình `.yml` (ví dụ: `custom_config.yml`)

### Bước 2: Chuyển đổi sang ONNX

```bash
python convert_transformer_split.py \
    --config <đường_dẫn_config> \
    --weights <đường_dẫn_weights> \
    --output_dir <thư_mục_output> \
    --verify
```

**Ví dụ:**

```bash
python convert_transformer_split.py \
    --config weights/custom_config_01102025.yml \
    --weights weights/weights/transformerocr.pth \
    --output_dir converted_weights_transformer/ \
    --verify
```

**Kết quả:**
```
✓ Saved to converted_weights_transformer/encoder.onnx
✓ Saved to converted_weights_transformer/decoder.onnx
✓ All outputs match!
```

### Bước 3: Chạy inference

```bash
python inference_onnx_transformer_split.py \
    --image <đường_dẫn_ảnh> \
    --config <đường_dẫn_config>
```

**Ví dụ:**

```bash
python inference_onnx_transformer_split.py \
    --image sample/35944.png \
    --config weights/custom_config_01102025.yml
```

**Kết quả:**
```
============================================================
Result: Mầm non: 141 thí sinh.
============================================================
Time: 0.15s
```

---

## Tham số dòng lệnh

### convert_transformer_split.py

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--config` | Đường dẫn file cấu hình YAML | (bắt buộc) |
| `--weights` | Đường dẫn file trọng số .pth | (bắt buộc) |
| `--output_dir` | Thư mục lưu file ONNX | `./converted_weights_transformer/` |
| `--verify` | Kiểm tra kết quả sau chuyển đổi | False |

### inference_onnx_transformer_split.py

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--image` | Đường dẫn ảnh cần nhận dạng | (bắt buộc) |
| `--config` | Đường dẫn file cấu hình YAML | (bắt buộc) |
| `--encoder` | Đường dẫn encoder.onnx | `./converted_weights_transformer/encoder.onnx` |
| `--decoder` | Đường dẫn decoder.onnx | `./converted_weights_transformer/decoder.onnx` |
| `--max_seq_length` | Độ dài tối đa của output | 128 |
| `--show_confidence` | Hiển thị độ tin cậy | False |

---

## Cấu trúc file ONNX

```
converted_weights_transformer/
├── encoder.onnx    # CNN + Transformer Encoder (~80MB)
└── decoder.onnx    # Transformer Decoder (~40MB)
```

---

## Yêu cầu cài đặt

```bash
pip install torch torchvision onnx onnxruntime pillow pyyaml numpy
```

Để chạy trên GPU:
```bash
pip install onnxruntime-gpu
```

---

## Lưu ý kỹ thuật

1. **Kiểm tra loại mô hình**: Xem trường `seq_modeling` trong file config
   - `seq_modeling: seq2seq` → Dùng script Seq2Seq
   - `seq_modeling: transformer` → Dùng script Transformer (hướng dẫn này)

2. **Verify model ONNX**:
   ```python
   import onnx
   model = onnx.load("encoder.onnx")
   onnx.checker.check_model(model)
   ```

3. **Inference nhanh hơn trên CPU**: Cài `onnxruntime` thay vì `onnxruntime-gpu` nếu không có GPU phù hợp.

---

## License

MIT

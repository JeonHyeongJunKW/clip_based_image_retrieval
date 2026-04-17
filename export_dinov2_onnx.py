"""
DINOv2 모델을 ONNX (FP16)로 변환하는 스크립트.

사용법:
    python3 export_dinov2_onnx.py --model-size base
    python3 export_dinov2_onnx.py --model-size large

출력:
    onnx_models/dinov2_vit{s|b|l}14_fp16.onnx
"""

import argparse
import os
import torch
from transformers import Dinov2Model


MODEL_MAP = {
    "small": "facebook/dinov2-small",   # 384-dim
    "base":  "facebook/dinov2-base",    # 768-dim
    "large": "facebook/dinov2-large",   # 1024-dim
}


class Dinov2Wrapper(torch.nn.Module):
    """CLS 토큰만 추출하여 L2 정규화."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(cls_token, p=2, dim=-1)


def export_dinov2_fp16(model_id, size_tag, output_dir):
    print(f"모델 로드: {model_id}")
    model = Dinov2Model.from_pretrained(model_id).eval()
    wrapper = Dinov2Wrapper(model).half().eval()

    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float16)
    if torch.cuda.is_available():
        wrapper = wrapper.cuda()
        dummy = dummy.cuda()

    output_path = os.path.join(output_dir, f"dinov2_vit{size_tag}14_fp16.onnx")
    print("ONNX 변환 중...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), output_path,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  -> {output_path} ({os.path.getsize(output_path) / (1024*1024):.1f} MB)")

    try:
        import onnx
        import onnxsim
        print("ONNX simplifier 적용 중...")
        model_simp, check = onnxsim.simplify(onnx.load(output_path))
        if check:
            onnx.save(model_simp, output_path)
            print(f"  단순화 완료: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        else:
            print("  단순화 검증 실패, 원본 유지")
    except ImportError:
        print("  onnxsim 미설치, 스킵")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(output_path))
        print("[검증] ONNX 모델 OK")
    except Exception as e:
        print(f"[검증] 실패: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="DINOv2 ONNX (FP16) Export")
    parser.add_argument("--model-size", choices=list(MODEL_MAP.keys()),
                        default="base", help="모델 크기 (기본: base)")
    parser.add_argument("--output-dir", default="onnx_models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    size_tag = args.model_size[0]  # s / b / l
    onnx_path = export_dinov2_fp16(MODEL_MAP[args.model_size], size_tag, args.output_dir)

    engine_name = f"dinov2_vit{size_tag}14_fp16.engine"
    print(f"\n다음 단계 - TensorRT 엔진 변환:")
    print(f"  /usr/src/tensorrt/bin/trtexec \\")
    print(f"      --onnx={onnx_path} \\")
    print(f"      --saveEngine={engine_name} \\")
    print(f"      --fp16")


if __name__ == "__main__":
    main()

# clip_based_image_retrieval

DINOv2 기반 이미지 검색 파이프라인 (Jetson + TensorRT FP16).

쿼리 이미지를 주면 데이터베이스 폴더 안에서 시각적으로 가장 비슷한 이미지를
Top-K로 찾아낸다. 임베딩 추출은 DINOv2(CLS 토큰 + L2 정규화), 유사도는 코사인.

## 구성

| 파일 | 역할 |
| --- | --- |
| `export_dinov2_onnx.py` | DINOv2 → ONNX (FP16) 변환 |
| `Dockerfile`            | ONNX 변환용 컨테이너 (Jetson L4T ML 기반) |
| `clip_retrieval.cu`     | TensorRT 엔진으로 임베딩 + Top-K 검색 (CUDA/C++) |
| `CMakeLists.txt`        | C++ 빌드 설정 |

## 파이프라인

```
PyTorch (HuggingFace) ── export_dinov2_onnx.py ──► ONNX(FP16)
                                                    │
                                                    │ trtexec
                                                    ▼
                                                  TensorRT .engine
                                                    │
                                                    ▼
                                             clip_retrieval (CUDA)
```

## 1. ONNX 변환

Jetson에서 파이썬 환경이 지저분해지기 쉬우므로 Docker 권장.

```bash
docker build -t dinov2-export .
docker run --rm --runtime=nvidia \
    -v "$(pwd)/onnx_models:/workspace/onnx_models" \
    dinov2-export
```

기본값은 `base` (768-dim). 다른 크기는 CMD 오버라이드:

```bash
docker run --rm --runtime=nvidia \
    -v "$(pwd)/onnx_models:/workspace/onnx_models" \
    dinov2-export \
    python3 export_dinov2_onnx.py --model-size base
```

| 크기 | 임베딩 차원 | 파라미터 |
| --- | --- | --- |
| small | 384  | ~22M  |
| base  | 768  | ~86M  |
| large | 1024 | ~300M |

출력: `onnx_models/dinov2_vit{s|b|l}14_fp16.onnx`

## 2. TensorRT 엔진 빌드

Jetson 호스트에서 직접 실행.

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=onnx_models/dinov2_vitb14_fp16.onnx \
    --saveEngine=dinov2_vitb14_fp16.engine \
    --fp16
```

입력 해상도는 224×224 고정, 배치는 dynamic이다.

## 3. C++ 검색 바이너리 빌드

```bash
mkdir -p build && cd build
cmake ..
make -j
```

요구사항: OpenCV, CUDA, TensorRT (`nvinfer`).

## 4. 테스트

### 4.1 기본 실행 (GUI)

```bash
./build/clip_retrieval \
    dinov2_vitb14_fp16.engine \
    /path/to/database_folder \
    /path/to/query_folder
```

- 데이터베이스 폴더의 모든 이미지를 임베딩 → 메모리 인덱스 구축
- 쿼리 폴더의 이미지를 한 장씩 순회하며 Top-K 매칭을 `imshow`로 표시
- 키: 아무 키 → 다음 쿼리, `ESC` 또는 `q` → 종료

### 4.2 헤드리스 (파일 저장)

ssh로 접속했거나 디스플레이가 없을 때:

```bash
./build/clip_retrieval \
    dinov2_vitb14_fp16.engine \
    /path/to/database_folder \
    /path/to/query_folder \
    --save-dir results/
```

결과는 `results/result_0000.jpg`, `result_0001.jpg` ... 로 저장된다.

### 4.3 주요 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--model <dinov2\|clip>` | `dinov2` | 전처리 정규화 (DINOv2 = ImageNet norm) |
| `--threshold <float>`    | `0.3`    | 코사인 유사도 최소값 |
| `--top-k <int>`          | `3`      | 매칭 결과 개수 |
| `--save-dir <path>`      | -        | 결과를 파일로 저장 (설정 시 imshow 비활성) |
| `--undistort`            | off      | 경로에 `v4l2` 포함 이미지를 fisheye 역왜곡 |

### 4.4 최소 동작 확인 (smoke test)

같은 폴더를 DB와 쿼리로 줘서 자기 자신이 1위로 나오는지 확인:

```bash
./build/clip_retrieval \
    dinov2_vitb14_fp16.engine \
    /path/to/images \
    /path/to/images \
    --save-dir results/ --top-k 1
```

각 쿼리의 Top-1이 자기 자신(similarity ≈ 1.0)이면 임베딩/검색 경로가 정상.

### 4.5 실행 결과 해석

콘솔 로그 예시:

```
입력 바인딩[0]: pixel_values, 224x224
출력 바인딩[1]: image_embeds, dim=768
[디버그] DB 이미지 간 유사도 (샘플 20장):
  min=0.2145, max=0.7321, avg=0.4012
Query [1/5]: query_001.jpg
  1. db_042.jpg (similarity: 0.8234)
  2. db_017.jpg (similarity: 0.7512)
  3. db_089.jpg (similarity: 0.6901)
```

DB 간 평균 유사도가 threshold와 너무 가까우면 의미 있는 매칭이 어렵다는
신호 — threshold를 올리거나 이미지 분포를 점검해야 한다.

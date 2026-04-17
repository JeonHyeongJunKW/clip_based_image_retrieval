FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

WORKDIR /workspace

RUN pip3 install --no-cache-dir --upgrade pip wheel && \
    pip3 install --no-cache-dir "setuptools<70"

RUN pip3 install --no-cache-dir --prefer-binary \
    "transformers==4.44.2" \
    "onnx==1.16.2" \
    "onnxsim"

COPY export_dinov2_onnx.py .

CMD ["python3", "export_dinov2_onnx.py", "--model-size", "base"]

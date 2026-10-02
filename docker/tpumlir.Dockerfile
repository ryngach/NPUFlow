# TPU-MLIR toolchain for ONNX -> cvimodel (CV181x) conversion.
# Base image digest and tpu_mlir version are pinned so that speed/accuracy
# numbers stay comparable between runs. 1.25 is the version that produced
# the December 2025 models.
FROM sophgo/tpuc_dev@sha256:d46a29a349f10c893fddc089b2433da5b8ef2a0bce52b81419064bf3e26a31a0
ARG TPU_MLIR_VERSION=1.25
RUN pip3 install --no-cache-dir tpu_mlir==${TPU_MLIR_VERSION}
WORKDIR /workspace

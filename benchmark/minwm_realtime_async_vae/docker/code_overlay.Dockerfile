ARG BASE_IMAGE

FROM ${BASE_IMAGE}
ARG SGLANG_GIT_SHA
LABEL org.opencontainers.image.revision=${SGLANG_GIT_SHA}
LABEL seedleap.minwm.image.layer="dual-model-code-overlay"
COPY python/sglang /opt/sglang/python/sglang
WORKDIR /opt/sglang

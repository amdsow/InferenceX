# Combined image for the InferenceX llm-d-vllm framework.
#
# Base = vllm/vllm-openai (vLLM with the OpenAI-compatible API server).
# We add the EPP, the routing-sidecar, and Envoy on top so every node in
# a SLURM allocation can play any role (prefill, decode, or coordinator)
# from a single image. DeepEP / NVSHMEM / GDRCopy are NOT bundled by
# this base; they are not used by the simple 1P+1D recipe
# (LWS_GROUP_SIZE=1 short-circuits the wide-EP NVSHMEM env in
# server.sh). Wide-EP recipes will need a base that ships them.
#
# Configs (epp-config.yaml, envoy.yaml, per-topology recipes) are NOT
# baked in. They are mounted at runtime by job.slurm so config-only
# iteration does not require an image rebuild. See
# benchmarks/multi_node/llm-d/job.slurm for the expected mount layout.

ARG VLLM_BASE=vllm/vllm-openai:v0.23.0
FROM ${VLLM_BASE}
# Force-upgrade nvidia-cutlass-dsl only when explicitly requested. The
# v0.22 base needed it because its bundled wheel lacked cute.arch.fmin
# for the DSV4 sparse-attention code; stock v0.23.0 (what upstream
# wide-ep-lws runs) ships a working cutlass, so we leave it untouched
# there. Build the v0.22 base with --build-arg UPGRADE_CUTLASS_DSL=1.
ARG UPGRADE_CUTLASS_DSL=0
RUN if [ "$UPGRADE_CUTLASS_DSL" = "1" ]; then \
        pip install --no-cache-dir --upgrade nvidia-cutlass-dsl; \
    fi


COPY --from=ghcr.io/llm-d/llm-d-router-endpoint-picker-dev:main \
       /app/epp /usr/local/bin/epp

COPY --from=ghcr.io/llm-d/llm-d-router-disagg-sidecar-dev:main \
       /app/pd-sidecar /usr/local/bin/pd-sidecar

COPY --from=envoyproxy/envoy:distroless-v1.33.2 \
     /usr/local/bin/envoy /usr/local/bin/

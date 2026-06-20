# Vendored benchmark binaries

## nyann-bench-linux-arm64

Static `linux/arm64` build of [nyann-bench](https://github.com/neuralmagic/nyann-bench),
the Go load generator the upstream wide-ep-lws GB200 guide used. Vendored here
because the GB200 compute nodes build/run via enroot only (no Go toolchain, no
guaranteed registry egress), and the repo is bind-mounted into the job at
`/workspace`, so a committed static binary is the most reliable delivery.

- Source image: `ghcr.io/neuralmagic/nyann-bench:latest`
- Image digest: `sha256:7df0e11d67f2d71371307c51df21ddc9ded803ffcd272403fef641dda22b3c1a`
- Extracted from `/nyann-bench` (the image is FROM scratch with this as ENTRYPOINT)
- ELF 64-bit aarch64, statically linked, stripped

Used by `benchmarks/multi_node/llm-d/server.sh` when `BENCH_TOOL=nyann-bench`.
To refresh: `docker create --platform linux/arm64 ghcr.io/neuralmagic/nyann-bench:latest`
then `docker cp <id>:/nyann-bench` here.

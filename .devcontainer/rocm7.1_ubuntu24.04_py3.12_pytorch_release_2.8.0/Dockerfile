# Use ROCm build container on Ubuntu 24.04 as base
FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0

# Copy uv into image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get full-upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


RUN uv pip install --no-cache-dir --upgrade pip setuptools wheel

# This is a supposed fix to help with gfx1151
# See: https://github.com/pytorch/pytorch/issues/164346
ENV HSA_OVERRIDE_GFX_VERSION="11.0.0"

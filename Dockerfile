FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.8.2
ARG PYTHON_VERSION=3.11.17
ARG PYTHON_SHA256=bfb74ad39efae27cda510f134ab408e00f9992c56851cfc0b1cdb5646da11599

ENV UV_INSTALL_DIR=/usr/local/bin \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:/opt/cpython/3.11.17/bin:/usr/local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates pkg-config \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libffi-dev liblzma-dev libncursesw5-dev libgdbm-dev libnss3-dev \
        libexpat1-dev uuid-dev tk-dev \
        libxml2 libglu1-mesa libvulkan1 vulkan-tools \
        libx11-6 libx11-xcb1 libxcb1 libxcb-cursor0 libxcb-icccm4 \
        libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
        libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 \
        libxkbcommon0 libxkbcommon-x11-0 libxrandr2 libxinerama1 libxcursor1 \
        libxi6 libxrender1 libxext6 libsm6 libice6 libglib2.0-0 libdbus-1-3 \
        libfontconfig1 libfreetype6 libopengl0 libegl1 libgles2 libgl1 \
        libstdc++6 libgcc-s1 libatomic1 libgomp1 libnss3 libnspr4 \
        libasound2 libpulse0 libxfixes3 libxdamage1 libxcomposite1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh \
    && uv --version | grep -F "${UV_VERSION}"

RUN set -eux; \
    curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
        -o /tmp/Python.tar.xz; \
    echo "${PYTHON_SHA256}  /tmp/Python.tar.xz" | sha256sum --check -; \
    mkdir -p /tmp/python-src; \
    tar -xJf /tmp/Python.tar.xz -C /tmp/python-src --strip-components=1; \
    cd /tmp/python-src; \
    ./configure --prefix=/opt/cpython/${PYTHON_VERSION} --with-ensurepip=install; \
    make -j"$(nproc)"; \
    make install; \
    /opt/cpython/${PYTHON_VERSION}/bin/python3.11 --version; \
    rm -rf /tmp/python-src /tmp/Python.tar.xz

WORKDIR /workspace/active_adaptation
COPY . /workspace/active_adaptation

RUN set -eux; \
    ldconfig -p | grep -q 'libxml2.so.2'; \
    ldconfig -p | grep -q 'libGLU.so.1'; \
    ldconfig -p | grep -q 'libvulkan.so.1'; \
    ldconfig -p | grep -q 'libX11.so.6'; \
    PYTHON_BIN="/opt/cpython/${PYTHON_VERSION}/bin/python3.11"; \
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync \
        --project venv/isaaclab \
        --python "$PYTHON_BIN" \
        --frozen \
        --no-dev; \
    /opt/venv/bin/python -c 'import importlib.metadata as m, sys, torch; assert sys.version_info[:3] == (3, 11, 17); assert m.version("isaaclab") == "2.3.2.post1"; assert m.version("isaacsim") == "5.1.0.0"; assert torch.__version__.split("+")[0] == "2.7.0" and torch.version.cuda == "12.8"; print("Python", sys.version.split()[0], "Isaac Lab", m.version("isaaclab"), "Isaac Sim", m.version("isaacsim"), "PyTorch", torch.__version__, "CUDA", torch.version.cuda)'

COPY docker/entrypoint.sh /usr/local/bin/active-adaptation-entrypoint
RUN chmod 0755 /usr/local/bin/active-adaptation-entrypoint

ENV AA_CACHE_DIR=/var/cache/active-adaptation/aa \
    XDG_CACHE_HOME=/var/cache/active-adaptation/xdg \
    WARP_CACHE_PATH=/var/cache/active-adaptation/warp \
    OMNI_KIT_CACHE_ROOT=/var/cache/active-adaptation/kit \
    OMNI_KIT_USER_CACHE=/var/cache/active-adaptation/kit \
    OMNI_KIT_USER_CONFIG=/var/cache/active-adaptation/kit-config \
    OMNI_KIT_USER_DATA=/var/cache/active-adaptation/kit-data \
    HOME=/var/cache/active-adaptation/home

ENTRYPOINT ["/usr/local/bin/active-adaptation-entrypoint"]
CMD ["bash"]

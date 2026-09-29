# =============================================================================
# STAGE 1: base
# ROS2 install + non-root user matching host UID/GID + locale/sudo setup,
# PLUS this project's ROS package deps (formerly Dockerfile.deps).
# Build once, tag, reuse. The committed docker-compose.yml builds only this
# stage — it's the shareable, project-portable image.
# =============================================================================
ARG ROS_DISTRO=jazzy
FROM osrf/ros:${ROS_DISTRO}-desktop AS base

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=${ROS_DISTRO} \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        locales \
        sudo \
        curl \
        ca-certificates \
        gnupg2 \
        lsb-release \
        software-properties-common \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# --- Non-root user, UID/GID passed at build time so bind-mounted files
#     created in the container are owned by you on the host, not root. ---
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME
ENV USERNAME=$USERNAME

RUN rosdep update --rosdistro $ROS_DISTRO || true
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# --- project-specific ROS packages ---
USER root

# ur_simulation_gz pulls in gz_ros2_control, joint_trajectory_controller,
# joint_state_broadcaster, ur_description, ur_controllers and xacro.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros-gz \
        ros-${ROS_DISTRO}-ur-simulation-gz \
    && rm -rf /var/lib/apt/lists/*

# --- uv + project Python environment ---
# uv runs the project with the system Python (3.12, matching ROS jazzy) so
# apt-installed ROS Python packages stay importable via --system-site-packages.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:${PATH}" \
    # make the bind-mounted source tree importable without an install step
    PYTHONPATH=/home/ros/ws/src \
    # source ROS for non-interactive shells (e.g. `docker compose exec ros2 python3 ...`)
    BASH_ENV=/opt/ros/${ROS_DISTRO}/setup.bash

WORKDIR /tmp/uv-cache
COPY pyproject.toml uv.lock ./
RUN uv venv --system-site-packages --python /usr/bin/python3 /opt/venv \
    && uv sync --frozen --no-install-project \
    && chown -R $USER_UID:$USER_GID /opt/venv \
    && rm -rf /tmp/uv-cache

USER $USERNAME
CMD ["bash"]

# =============================================================================
# STAGE 2: dev
# Editor, LSPs, shell QoL, build tooling — everything you want in every ROS2
# project regardless of robot/sim target. Built on top of base+deps.
# Only the docker-compose override (uncommitted) builds this stage.
# =============================================================================
FROM base AS dev

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        tmux \
        zsh \
        ripgrep \
        fd-find \
        fzf \
        unzip \
        wget \
        build-essential \
        gdb \
        clang \
        clang-tools \
        clang-format \
        clangd \
        python3 \
        python3-venv \
        pipx \
        default-jre-headless \
        xterm \
        x11-apps \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_HOME=/usr/local/cargo
ENV PATH="${CARGO_HOME}/bin:${PATH}"

RUN mkdir -p ${CARGO_HOME} \
    && curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash \
    && cargo-binstall --no-confirm tree-sitter-cli \
    && chmod -R a+rX ${CARGO_HOME}

# --- Neovim, latest stable, glibc build (matches the Ubuntu base image) ---
ARG NVIM_ARCH=linux-x86_64
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-${NVIM_ARCH}.tar.gz \
    && tar -C /opt -xzf nvim-${NVIM_ARCH}.tar.gz \
    && ln -s /opt/nvim-${NVIM_ARCH}/bin/nvim /usr/local/bin/nvim \
    && rm nvim-${NVIM_ARCH}.tar.gz

ENV PIPX_BIN_DIR=/usr/local/bin
ENV PIPX_HOME=/opt/pipx

# - ruff: Ultra-fast Python linter and code formatter used by LazyVim conform/lint
# - debugpy: Python DAP adapter backend used by nvim-dap-python
RUN pipx install ruff debugpy basedpyright cmake-language-server

# --- Node-based language servers (Pyright & YAML) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    && npm install -g --no-fund --no-audit \
    yaml-language-server

# --- lemminx (XML LSP — covers package.xml, URDF/xacro) ---
RUN LEMMINX_VERSION=$(curl -s https://api.github.com/repos/eclipse-lemminx/lemminx/releases/latest | grep -oP '"tag_name": "\K[^"]+') \
    && mkdir -p /opt/lemminx \
    && curl -L "https://download.eclipse.org/lemminx/releases/${LEMMINX_VERSION}/org.eclipse.lemminx-uber.jar" -o /opt/lemminx/lemminx-uber.jar \
    && printf '#!/bin/sh\nexec java -jar /opt/lemminx/lemminx-uber.jar "$@"\n' > /usr/local/bin/lemminx \
    && chmod +x /usr/local/bin/lemminx

# --- codelldb (DAP adapter, self-contained — bundles its own LLDB) ---
RUN curl -L "https://github.com/vadimcn/codelldb/releases/latest/download/codelldb-linux-x64.vsix" -o /tmp/codelldb.vsix \
    && mkdir -p /opt/codelldb \
    && unzip -q /tmp/codelldb.vsix -d /opt/codelldb \
    && chmod +x /opt/codelldb/extension/adapter/codelldb \
    && rm /tmp/codelldb.vsix

USER $USERNAME
CMD ["bash"]

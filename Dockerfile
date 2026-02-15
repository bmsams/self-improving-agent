FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# Configure UV for container environment
ENV UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=1

# System deps (git is required by the self-improving agent).
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
# Install from requirements file
RUN uv pip install -r requirements.txt




RUN uv pip install aws-opentelemetry-distro>=0.10.1


# Set AWS region environment variable

ENV AWS_REGION=us-east-1
ENV AWS_DEFAULT_REGION=us-east-1


# Signal that this is running in Docker for host binding logic
ENV DOCKER_CONTAINER=1

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore

EXPOSE 8080
EXPOSE 8000

# Copy entire project (respecting .dockerignore)
COPY . .

# Ensure the agent user can modify the working tree.
RUN chown -R bedrock_agentcore:bedrock_agentcore /app

USER bedrock_agentcore

# Initialize a git repo inside the container (the build context excludes `.git/`).
RUN git config --global user.name "Self-Improving Agent" && \
    git config --global user.email "agent@self-improving.ai" && \
    git config --global init.defaultBranch main && \
    git init && git add -A && git commit -m "Initial commit in container"

CMD ["opentelemetry-instrument", "python", "-m", "runtime_entrypoint"]

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY config/ config/
COPY main.py .
COPY runtime_entrypoint.py .
COPY CLAUDE.md .
COPY README.md .
COPY Makefile .

# Install the package with AgentCore dependencies
RUN pip install --no-cache-dir -e ".[dev,agentcore]" && \
    pip install --no-cache-dir aws-opentelemetry-distro==0.10.1

# Configure git for the agent
RUN git config --global user.name "Self-Improving Agent" && \
    git config --global user.email "agent@self-improving.ai" && \
    git config --global init.defaultBranch main

# Initialize git repo (needed for agent operations)
RUN git init && git add -A && git commit -m "Initial commit in container"

# Create non-root user for AgentCore Runtime
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

# AgentCore Runtime listens on port 8080
EXPOSE 8080

# Health check for AgentCore Runtime
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/ping || exit 1

# Default: run via AgentCore Runtime with OpenTelemetry instrumentation
# Override with CMD ["--help"] for CLI mode
ENTRYPOINT ["opentelemetry-instrument", "python", "-m", "runtime_entrypoint"]

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY config/ config/
COPY main.py .
COPY CLAUDE.md .
COPY README.md .
COPY Makefile .

# Install the package and dev dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Configure git for the agent
RUN git config --global user.name "Self-Improving Agent" && \
    git config --global user.email "agent@self-improving.ai" && \
    git config --global init.defaultBranch main

# Initialize git repo (needed for agent operations)
RUN git init && git add -A && git commit -m "Initial commit in container"

# Default entrypoint
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]

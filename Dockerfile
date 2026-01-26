FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
COPY . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the streamlit application
CMD ["sh", "-c", "uv run streamlit run 首頁.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]

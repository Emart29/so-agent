# A single image that runs the tool, the tests, and the report build.
#
# No services to orchestrate: everything the project stores lives in SQLite and
# two JSON files, and the only network dependency is whichever provider's API
# key is supplied at run time.

FROM python:3.11-slim

# Fail fast and log in real time. A benchmark run is long, and buffered output
# makes a stalled run indistinguishable from a slow one.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Runs as a non-root user. The container writes only to /app/data, which is
# where the attempt log and the generated report go.
RUN useradd --create-home --uid 1000 runner \
    && mkdir -p /app/data \
    && chown -R runner:runner /app
USER runner

ENV LOG_DB_PATH=/app/data/runs.db

ENTRYPOINT ["so-agent"]
CMD ["providers"]

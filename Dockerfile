FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /artifact
COPY . /artifact

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --no-deps -e .

# Default: fast end-to-end check that also regenerates every figure into ./figures
CMD ["./run_experiments.sh", "smoke_test"]

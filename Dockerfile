FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /artifact
COPY . /artifact

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --no-deps -e .

# Default: regenerate every paper figure/table from the bundled results into ./figures
CMD ["./run_experiments.sh", "figures"]

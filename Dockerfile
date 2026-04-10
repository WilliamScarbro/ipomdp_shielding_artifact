FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /artifact

COPY . /artifact

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --no-deps -e .

CMD ["python", "scripts/smoke_test.py"]

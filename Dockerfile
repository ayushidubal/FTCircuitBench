FROM python:3.12-slim-bookworm

WORKDIR /app
COPY . /app
RUN pip install --upgrade pip \
    && pip install --prefer-binary -e . \
    && pip install qiskit_qasm3_import
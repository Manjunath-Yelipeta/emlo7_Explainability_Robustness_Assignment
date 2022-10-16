FROM python:3.9-slim

ENV GRADIO_SERVER_PORT 8080

WORKDIR /opt/src

COPY requirements.txt .

RUN pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY . .

EXPOSE 8080

ENTRYPOINT ["python3","demo_scripted.py"]
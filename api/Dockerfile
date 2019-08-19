FROM python:3
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir scripts
RUN mkdir models

COPY ./scripts/models.py ./scripts
COPY ./models/generator.pt ./models
COPY ./index.py .

CMD ["python", "index.py"]

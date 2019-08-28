FROM python:3
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=index.py
ENV FLASK_DEBUG=0
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]

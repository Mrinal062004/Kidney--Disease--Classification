FROM python:3.8-slim-bluster

RUN apt update -y && apt install awscli -y
WORKDIR /app
COPY . /app
RUN pip intall -requirements.txt
CMD ["python3","app.py"]

FROM python:3.9-slim-buster
WORKDIR /dlwpt-code
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY ./p2ch11 ./p2ch11
COPY ./util ./util
CMD [ "python3", "./p2ch11/prepcache.py"]
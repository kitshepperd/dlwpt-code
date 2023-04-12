FROM python:3.9-slim-buster
WORKDIR /dlwpt-code
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY ./p2ch11 ./p2ch11
CMD [ "python3", "./p2ch11/disk_cache_play/hello_world.py"]
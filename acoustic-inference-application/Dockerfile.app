# pull official base image
FROM python:3.6-slim-buster

COPY acoustic-inference-application/requirements_app.txt .

# install system dependencies
RUN apt-get update \
  && apt-get -y install netcat gcc \
  && apt-get clean

# add and install requirements
#COPY ./requirements.txt .
RUN pip install -r requirements_app.txt

RUN adduser -disabled-password acoustic_app

# set working directory
WORKDIR /home/acoustic_app

#RUN mkdir -p /usr/src
#WORKDIR /usr/src

#RUN useradd --disabled-password myuser
#RUN useradd myuser

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# add app
#COPY . .
#COPY acoustic-inference-application .
#RUN mkdir -p /resources
#RUN mkdir -p /data


COPY resources/ ../resources

RUN pip install -e ../resources/vs_db_maintainer/
RUN pip install -e ../resources/vs_data_query/
RUN pip install -e ../resources/sqlite_resources/
RUN pip install -e ../resources/data_types/

RUN apt-get install net-tools

#COPY acoustic-inference-application/boot.sh ../

#RUN chmod +x /home/acoustic_app/boot.sh

#RUN chown -R .:myuser ./
#USER myuser

# add and run as non-root user

# run server
#CMD python manage.py run -h 0.0.0.0
#CMD flask run --host=0.0.0.0
#ENTRYPOINT ["./acoustic-inference-application/boot.sh"]
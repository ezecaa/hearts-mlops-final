#Ubicación: ./dockerfiles/postgres/dockerfile 
# Use the official PostgreSQL image
FROM postgres:latest

COPY healthcare-dataset-stroke-data.csv /docker-entrypoint-initdb.d/
#COPY healthcare-dataset-stroke-data.csv /docker-entrypoint-initdb.d/
#COPY mlflow_stroke.sql /docker-entrypoint-initdb.d/
# Copy the initialization script into the container
COPY mlflow.sql /docker-entrypoint-initdb.d/
# Copy el dataset en la imagen del postgres 

#COPY healthcare-dataset-stroke-data.csv /docker-entrypoint-initdb.d/
COPY stroke.sql /docker-entrypoint-initdb.d/
#COPY tablastrok.sql /docker-entrypoint-initdb.d/

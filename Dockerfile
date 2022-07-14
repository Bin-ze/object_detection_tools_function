# Container image that runs your code
FROM ubuntu:latest

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY . /

WORKDIR /

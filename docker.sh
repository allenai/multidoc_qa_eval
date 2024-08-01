#!/bin/bash


sudo docker build -t mdqa -f "./Dockerfile" .


sudo docker run --env-file .env --rm -it -v "$(pwd)":/app --workdir /app mdqa bash

#!/bin/sh

. ./.env

sed -e 's/"//g' -e 's/: /=/' .env | while read line; do export $line; done

cd src/

uvicorn main:app --host "0.0.0.0" --port 8080
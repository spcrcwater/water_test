#!/bin/bash

wget -q --spider http://google.com

cicomid=0
if [ $? -eq 0 ]; then
    cd /home/pi/Desktop/waterspcrc/Ph-03/
    git pull origin main
    cicomid_new=$(git rev-parse HEAD)
    if[ cicomid -ne cicomid_new]; then
      sudo systemctl restart codetest.service
    else
      echo 'No changes'
    fi

else
    echo 'No Internet'
fi
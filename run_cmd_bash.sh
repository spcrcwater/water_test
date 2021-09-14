#!/bin/bash

wget -q --spider http://google.com

if [ $? -eq 0 ]; then
    cd /home/pi/Desktop/waterspcrc/Ph-03/
    git pull 
else
    echo 'No Internet'
fi
#!/bin/bash
while :
do
wget -q --spider http://google.com

if [ $? -eq 0 ]; then
    git pull 
    python3 test.py
else
    python3 test.py
fi
done

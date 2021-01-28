#! /bin/bash

file=()
name=$HOME"/data"

echo "search name: $name"

for entry in "$HOME"/*
do 
    files+=("$entry")
done



if [ -e "$name" ]; then
    :
else
    echo "data folder does not exist"
    exit 1
fi


if [ -e "$name/log" ]; then
    echo "folder setup sucessful"
else
    mkdir -p "$name/log"
    echo " folder setup sucessful"
fi

echo 
echo "installing dependencies"
echo 
pip install -r hotel_cloud/requirement.txt

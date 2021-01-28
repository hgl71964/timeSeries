#! /bin/bash

file=()
name=$HOME"/data"

echo "search name: $name"

for entry in "$HOME"/*
do 
    files+=("$entry")
done



if [ -e "$name" ]; then
    echo "File exists"
else
    echo "File does not exist"
    exit 1
fi


if [ -e "$name/log" ]; then
    echo "setup sucessful"
    exit 0
else
    mkdir -p "$name/log"
    echo "setup sucessful"
    exit 0
fi

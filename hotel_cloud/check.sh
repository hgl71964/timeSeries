#! /bin/bash

file=()

# one-liner to get script's pwd
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/data

if [ -e "$DIR/log" ]; then
    echo "log folder setup successful"
else
    mkdir -p "$DIR/log"
    echo "set up log folder"
fi


echo
echo "installing dependencies"
PARENT=$(dirname $DIR)

pip install -r "$PARENT"/requirement.txt

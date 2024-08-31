#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <virtual_env_root_directory>"
    exit 1
fi

DEST_DIR=$1

cp build/package_modifications/parser.py $DEST_DIR/lib/python3.8/dist-packages/adblockparser/parser.py
cp build/package_modifications/webdriver.py $DEST_DIR/lib/python3.8/dist-packages/selenium/webdriver/remote/webdriver.py
cp build/package_modifications/debugger.py $DEST_DIR/lib/python3.8/dist-packages/selenium/webdriver/common/devtools/v111/debugger.py
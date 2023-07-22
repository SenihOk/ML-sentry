#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d venv ]
then
    python3 -m venv venv
fi
source venv/bin/activate

pip install tensorflow opencv-python numpy screeninfo

python3 minimain.py
#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d "d2a2d" ]; then
    python3 -m venv d2a2d
fi

# Activate the virtual environment
source d2a2d/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Optionally, run the Python script (uncomment if needed)
# python your_script.py
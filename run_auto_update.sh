#!/bin/bash

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate streamlit

# Navigate to the directory containing the script and the .git repository
cd /tmp/bmss

# Run the Python script
python update_database.py


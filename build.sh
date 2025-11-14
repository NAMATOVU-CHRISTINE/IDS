#!/bin/bash
# Render build script - trains model during deployment

echo "Installing dependencies..."
pip install -r requirements_api.txt

echo "Training model..."
python train_and_save_model.py

echo "Build complete!"

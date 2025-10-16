#!/bin/bash

# Exit immediately if a command fails
set -e

# Create a new conda environment and activate it
conda create -n DKdMRI python==3.12 -y
source activate DKdMRI

# Install necessary packages via conda
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install nvidia/label/cuda-12.4.1::cuda-toolkit -y
conda install conda-forge::cudnn -y

# Install additional packages via pip
pip install monai==1.4 fvcore medpy dipy

# Clone nnUNet repository and install it
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

# Go back to the original directory
cd ..

# clone Conversion repository and install it
git clone https://github.com/pnlbwh/conversion.git
cd conversion
pip install . 

# Go back to the original directory
cd ..

# Run the move_files.py script
python src/move_files.py
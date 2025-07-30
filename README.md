Local AI Image Generator
A Python script to generate high-quality images from text prompts locally on your own computer. This program uses the powerful Stable Diffusion XL model and runs on your NVIDIA GPU, ensuring privacy and unlimited free generations.

Features
Interactive Prompt: Run the script once and generate as many images as you want without reloading.

Local & Private: No data is sent to the cloud. All generation happens on your machine.

High Quality: Utilizes the Stable Diffusion XL 1.0 base model for excellent image quality.

Simple to Use: A straightforward command-line interface.

Requirements
Hardware
GPU: A powerful NVIDIA GPU with CUDA support and at least 8 GB of VRAM is required. Recommended GPUs include the RTX 30-series (3060, 3070, etc.) and RTX 40-series (4070, 4090, etc.).

Disk Space: At least 20 GB of free disk space to download and cache the AI model.

Software
Python 3.10+

Git

Setup & Installation
Follow these steps in your terminal to set up the project.

1. Clone the Repository
First, clone this repository to your local machine.

git clone https://github.com/YourUsername/YourRepositoryName.git
cd YourRepositoryName

(Replace YourUsername and YourRepositoryName with your details)

2. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep dependencies isolated.

# Create the environment
python -m venv .venv

# Activate the environment (on Windows)
.\.venv\Scripts\activate

3. Install PyTorch with CUDA
This is the most important step. It installs the core library that allows Python to use your NVIDIA GPU.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

4. Install Other Required Libraries
Install the remaining Python libraries needed for the script.

pip install diffusers transformers accelerate safetensors

How to Use
1. Run the Script
With your virtual environment activated, run the main script.

python local_generator.py

2. Initial Model Download
The first time you run the script, it will download the Stable Diffusion XL model (approx. 12 GB). This is a one-time process and may take a while depending on your internet connection. Please be patient.

3. Enter Your Prompt
Once the model is loaded, you will see the following message:

--- Model Loaded Successfully! Ready to generate. ---

Enter your prompt (or type 'exit' to quit):

Type any idea you have for an image and press Enter.

4. Generate and Save
The script will generate the image and save it as a .png file in the project directory. The filename will be based on your prompt.

5. Continue or Exit
The script will loop back and ask for a new prompt. You can continue creating as many images as you like. To stop the program, simply type exit and press Enter.

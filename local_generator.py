import torch
from diffusers import DiffusionPipeline
import os
import re

# --- Configuration ---
# This is the ID of the model we want to use.
# Stable Diffusion XL is a powerful model that works great on your RTX 4090.
model_id = "stabilityai/stable-diffusion-xl-base-1.0"


def create_safe_filename(prompt):
    """
    Creates a filesystem-safe filename from a text prompt to avoid errors.
    """
    # Keep only letters, numbers, and spaces from the prompt.
    text = re.sub(r'[^a-zA-Z0-9 ]', '', prompt)
    # Take the first 50 characters to keep filenames from getting too long.
    text = text[:50]
    # Replace spaces with underscores and add the .png extension.
    return text.replace(' ', '_') + ".png"


def main():
    """
    The main function to set up the pipeline and run the interactive loop.
    """
    print("Initializing the generation pipeline...")
    print("This will take a moment as the model is loaded into the GPU.")

    # Check if a CUDA-enabled GPU is available.
    if not torch.cuda.is_available():
        print("Error: PyTorch cannot find a CUDA-enabled GPU. Please check your installation.")
        return

    # 1. Load the model pipeline only ONCE.
    # This is the slow part. We do it outside the loop.
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        # Move the pipeline to the GPU to use your RTX 4090.
        pipe.to("cuda")
        print("\n--- Model Loaded Successfully! Ready to generate. ---")
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("Please ensure you have an internet connection for the first run and sufficient disk space.")
        return

    # 2. Start the interactive loop to get prompts from the user.
    while True:
        # Get user input.
        prompt = input("\nEnter your prompt (or type 'exit' to quit): ")

        # Check for the exit command.
        if prompt.lower().strip() == 'exit':
            print("Exiting program.")
            break

        # Check if the user just pressed Enter without typing anything.
        if not prompt:
            print("Please enter a prompt.")
            continue

        print(f"\nGenerating image for: '{prompt}'")
        print("This may take a moment...")

        try:
            # 3. Generate the image using the pre-loaded pipeline.
            # This is the part that uses your GPU's power.
            image = pipe(prompt=prompt).images[0]

            # 4. Save the image with a unique filename.
            output_filename = create_safe_filename(prompt)
            image.save(output_filename)

            full_path = os.path.abspath(output_filename)
            print(f"\nSuccessfully saved image to: {full_path}")

        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")


if __name__ == "__main__":
    main()

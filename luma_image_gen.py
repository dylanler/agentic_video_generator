import os
import requests
import time
from dotenv import load_dotenv
from lumaai import LumaAI

# Load environment variables
load_dotenv()
LUMAAI_API_KEY = os.getenv('LUMAAI_API_KEY')

# Initialize Luma AI client
client = LumaAI()

def generate_image(prompt, output_dir="generated_images"):
    """
    Generate an image using Luma AI based on the given prompt.
    
    Args:
        prompt (str): The text prompt describing the image to generate
        output_dir (str): Directory to save the generated image (default: 'generated_images')
        
    Returns:
        tuple: (image_url, filepath) - URL of the generated image and path to the saved file
               Returns (None, None) if generation fails
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Start the generation
        generation = client.generations.image.create(
            prompt=prompt,
        )
        
        # Wait for completion
        completed = False
        print("Starting image generation...")
        while not completed:
            generation = client.generations.get(id=generation.id)
            if generation.state == "completed":
                completed = True
            elif generation.state == "failed":
                print(f"Generation failed: {generation.failure_reason}")
                return None, None
            print("Dreaming...")
            time.sleep(2)
        
        # Get the image URL
        image_url = generation.assets.image
        if not image_url:
            print("No image URL in generation response")
            return None, None
            
        # Create a filename with the generation ID
        filename = f"{generation.id}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Download the image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(filepath, 'wb') as file:
            file.write(response.content)
            
        if not os.path.exists(filepath):
            print(f"Failed to save image to {filepath}")
            return None, None
            
        print(f"Image successfully generated and saved to: {filepath}")
        return image_url, filepath
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Example usage
    prompt = "A cute robot painting a sunset landscape"
    try:
        image_url, image_path = generate_image(prompt)
        if image_url and image_path:
            print(f"Generated image URL: {image_url}")
            print(f"Generated image saved at: {image_path}")
        else:
            print("Failed to generate image")
    except Exception as e:
        print(f"Failed to generate image: {str(e)}")

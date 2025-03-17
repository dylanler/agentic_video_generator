import os
import requests
import fal_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_image(prompt, output_dir="generated_images"):
    """
    Generate an image using Fal AI based on the given prompt.
    
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
        
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(log["message"])
        
        # Start the generation
        print("Starting image generation...")
        result = fal_client.subscribe(
            "fal-ai/sana",
            arguments={
                "prompt": prompt
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        
        if not result or 'images' not in result or not result['images']:
            print("No image data in generation response")
            return None, None
            
        # Get the image URL from the result
        image_url = result['images'][0]
        
        # Create a unique filename using timestamp
        import time
        filename = f"fal_gen_{int(time.time())}.png"
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

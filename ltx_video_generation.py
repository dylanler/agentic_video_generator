"""
LTX Video Generation using Fal AI

This module provides functionality to generate videos using Fal AI's LTX models.
It supports both text-to-video and image-to-video generation.

Installation Requirements:
    pip install fal-client python-dotenv

Environment Setup:
    1. Create a .env file in your project root
    2. Add your Fal AI API key: FAL_API_KEY=your_api_key_here

Usage Examples:

1. Text-to-Video Generation:
    from ltx_video_generation import generate_ltx_video
    
    # Basic text-to-video
    result = generate_ltx_video(
        prompt="A cinematic scene of a sunset over mountains"
    )

2. Image-to-Video Generation:
    # Image-to-video with a reference image
    result = generate_ltx_video(
        prompt="Make this image dynamic with a slow camera pan",
        image_url="https://your-image-url.com/image.jpg"
    )

3. Advanced Usage with Additional Parameters:
    # Using custom model arguments
    result = generate_ltx_video(
        prompt="Your prompt here",
        image_url="optional_image_url",
        model_args={
            "custom_param": "value"
        }
    )

The function returns a dictionary containing the generated video information.
Progress logs will be printed during generation.

Note: The generation process can take several minutes depending on the complexity
of the prompt and server load.
"""

import os
import fal_client
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Union, Dict, Any

# Load environment variables
load_dotenv()

def download_video(url: str, output_path: str) -> str:
    """
    Download a video from a URL to a specified path.
    
    Args:
        url (str): The URL of the video to download
        output_path (str): The path where the video should be saved
    
    Returns:
        str: The path where the video was saved
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Download the video
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return output_path

def on_queue_update(update):
    """Callback function to handle queue updates and print progress logs."""
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

def generate_ltx_video(
    prompt: str,
    image_url: Optional[str] = None,
    output_path: Optional[str] = None,
    model_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a video using Fal AI's LTX models, supporting both image-to-video and text-to-video generation.
    
    Args:
        prompt (str): The text description of the desired video.
        image_url (Optional[str]): URL of the input image for image-to-video generation.
                                 If None, text-to-video generation will be used.
        output_path (Optional[str]): Path where the generated video should be saved.
                                   If None, only the URL will be returned.
        model_args (Optional[Dict[str, Any]]): Additional model arguments to pass to the API.
    
    Returns:
        Dict[str, Any]: The API response containing the generated video information.
                       If output_path is provided, also includes 'saved_path' key.
                       Response includes: seed, video.url, video.file_name, video.file_size
    """
    # Set up the API key
    fal_api_key = os.getenv("FAL_API_KEY")
    if not fal_api_key:
        raise ValueError("FAL_API_KEY not found in environment variables")
    
    # Prepare base arguments
    arguments = {
        "prompt": prompt,
    }
    
    # Add image URL if provided
    if image_url:
        arguments["image_url"] = image_url
        model_endpoint = "fal-ai/ltx-video/image-to-video"
    else:
        model_endpoint = "fal-ai/ltx-video"
    
    # Add any additional model arguments
    if model_args:
        arguments.update(model_args)
    
    try:
        # Make the API call
        result = fal_client.subscribe(
            model_endpoint,
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        
        # Extract video URL from the response
        video_url = result.get('video', {}).get('url')
        
        if not video_url:
            raise ValueError("No video URL found in the API response")
            
        # Add video URL to the top level of result for backward compatibility
        result['video_url'] = video_url
        
        # Download the video if output path is provided
        if output_path:
            saved_path = download_video(video_url, output_path)
            result['saved_path'] = saved_path
            print(f"Video saved to: {saved_path}")
        
        print(f"Video URL: {video_url}")
        print(f"Video file name: {result.get('video', {}).get('file_name')}")
        print(f"Video file size: {result.get('video', {}).get('file_size')} bytes")
        
        return result
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    """
    Example 1: Text-to-Video Generation
    ---------------------------------
    # Get just the URL
    result = generate_ltx_video(
        prompt="A serene mountain lake at sunset, with gentle ripples on the water..."
    )
    
    # Download to a specific path
    result = generate_ltx_video(
        prompt="A serene mountain lake at sunset...",
        output_path="generated_videos/timestamp/lake_sunset_video.mp4"
    )
    
    Example 2: Image-to-Video Generation
    ---------------------------------
    # Get just the URL
    result = generate_ltx_video(
        prompt="Transform this image into a dynamic scene...",
        image_url="https://example.com/your-image.jpg"
    )
    
    # Download to a specific path
    result = generate_ltx_video(
        prompt="Transform this image into a dynamic scene...",
        image_url="https://example.com/your-image.jpg",
        output_path="generated_videos/timestamp/transformed_video.mp4"
    )
    """
    
    # Example: Text-to-video generation
    text_prompt = """A serene mountain lake at sunset, with gentle ripples on the water
                    reflecting the orange and purple sky. A small wooden boat gently
                    drifts across the frame from left to right."""
    
    # Get just the URL
    text_result = generate_ltx_video(prompt=text_prompt)
    
    # Download to a specific path
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"generated_videos/{timestamp}"
    
    text_result_with_download = generate_ltx_video(
        prompt=text_prompt,
        output_path=f"{output_dir}/lake_sunset_video.mp4"
    )

import os
import requests
import time
from lumaai import LumaAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_all_generations(limit=100, offset=0):
    """
    Retrieves all generations from Luma AI with pagination support.
    
    Args:
        limit (int): Maximum number of generations to retrieve per request. Default is 100.
        offset (int): Number of generations to skip. Default is 0.
        
    Returns:
        GenerationListResponse: Response object containing generations
    """
    try:
        # Initialize the Luma AI client
        client = LumaAI(
            auth_token=os.environ.get("LUMAAI_API_KEY")
        )
        
        # Get the list of generations
        generations = client.generations.list(limit=limit, offset=offset)
        
        return generations
    except Exception as e:
        print(f"Error retrieving generations: {str(e)}")
        return None

def get_generation_by_id(generation_id):
    """
    Retrieves a specific generation by its ID.
    
    Args:
        generation_id (str): The ID of the generation to retrieve
        
    Returns:
        object: The generation object if found, None otherwise
    """
    try:
        # Initialize the Luma AI client
        client = LumaAI(
            auth_token=os.environ.get("LUMAAI_API_KEY")
        )
        
        # Get the generation by ID
        generation = client.generations.get(id=generation_id)
        
        return generation
    except Exception as e:
        print(f"Error retrieving generation {generation_id}: {str(e)}")
        return None

def delete_generation(generation_id):
    """
    Deletes a specific generation by its ID.
    
    Args:
        generation_id (str): The ID of the generation to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        # Initialize the Luma AI client
        client = LumaAI(
            auth_token=os.environ.get("LUMAAI_API_KEY")
        )
        
        # Delete the generation
        client.generations.delete(id=generation_id)
        
        return True
    except Exception as e:
        print(f"Error deleting generation {generation_id}: {str(e)}")
        return False

if __name__ == "__main__":
        # Initialize the Luma AI client
    client = LumaAI(
        auth_token=os.environ.get("LUMAAI_API_KEY")
    )

    # Get the list of supported camera motions
    supported_camera_motions = client.generations.camera_motion.list()
    print(f"Supported camera motions: {supported_camera_motions}")
    # Example usage
    generations = get_all_generations(limit=100)
    #print(f"Retrieved generations: {generations}")
    
    # Print details of each generation
    for gen in generations.generations:
        print(f"ID: {gen.id}, Created at: {gen.created_at}")
        if hasattr(gen, "assets") and gen.assets and hasattr(gen.assets, "video"):
            print(f"Video URL: {gen.assets.video}")

import os
import json
from datetime import datetime
import anthropic
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
from luma_image_gen import generate_image

load_dotenv()

def generate_single_image(args):
    env_idx, prompt_data, output_dir = args
    try:
        env_dir = os.path.join(output_dir, f"environment_{env_idx}")
        os.makedirs(env_dir, exist_ok=True)
        
        image_path = generate_image(
            prompt_data["prompt_text"],
            output_dir=env_dir
        )
        return {
            "environment_index": env_idx,
            "prompt_number": prompt_data["prompt_number"],
            "image_path": image_path
        }
    except Exception as e:
        print(f"Error generating image for environment {env_idx}, prompt {prompt_data['prompt_number']}: {str(e)}")
        return None

class SceneEnvironmentGenerator:
    def __init__(self, video_dir):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_dir = video_dir
        
    def generate_environment_prompts(self, scenes):
        """Generate 10 prompts for each unique physical environment using Claude."""
        # First, identify unique environments with 1-based indexing
        unique_environments = {}
        for scene in scenes:
            env_desc = scene['scene_physical_environment']
            if env_desc not in unique_environments:
                unique_environments[env_desc] = len(unique_environments) + 1  # Changed to 1-based indexing
        
        prompt = """
        For each physical environment description, create exactly 10 different prompts that:
        1. Retain the core physical environment description
        2. Add variations of camera angles (e.g., wide shot, close-up, aerial view, etc.)
        3. Include different character positions and interactions
        4. Describe key items and their placement
        
        Return the prompts in this JSON format:
        {
            "environment_index": integer,
            "environment_description": "original description",
            "prompts": [
                {
                    "prompt_number": integer,  # Must be between 1 and 10
                    "prompt_text": "detailed prompt with camera angle and character placement"
                }
            ]
        }
        
        The prompts array MUST contain exactly 10 items, numbered from 1 to 10.
        """
        
        all_prompts = []
        for env_desc, env_idx in unique_environments.items():
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.7,
                system="You are an expert at creating detailed image generation prompts.",
                messages=[{
                    "role": "user",
                    "content": f"Environment description: {env_desc}\n\n{prompt}"
                }]
            )
            
            try:
                env_prompts = json.loads(response.content[0].text)
                # Ensure exactly 10 prompts
                if len(env_prompts["prompts"]) != 10:
                    print(f"Warning: Environment {env_idx} has {len(env_prompts['prompts'])} prompts, truncating to 10")
                    env_prompts["prompts"] = env_prompts["prompts"][:10]
                env_prompts["environment_index"] = env_idx  # Ensure correct environment index
                all_prompts.append(env_prompts)
            except Exception as e:
                print(f"Error parsing prompts for environment {env_idx}: {str(e)}")
                continue
        
        # Save prompts to file in video directory
        output_path = os.path.join(self.video_dir, f'scene_physical_environment_prompts_{self.timestamp}.json')
        with open(output_path, 'w') as f:
            json.dump(all_prompts, f, indent=2)
            
        return all_prompts, output_path
    
    def generate_environment_images(self, prompts_data):
        """Generate images for all prompts using multiprocessing."""
        output_dir = os.path.join(self.video_dir, "environment_images")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare arguments for parallel processing
        generation_args = []
        for env_data in prompts_data:
            env_idx = env_data["environment_index"]
            for prompt in env_data["prompts"]:
                generation_args.append((env_idx, prompt, output_dir))
        
        # Generate images in parallel
        results = []
        with ProcessPoolExecutor() as executor:
            for result in executor.map(generate_single_image, generation_args):
                if result:
                    results.append(result)
        
        # Save image generation results
        results_path = os.path.join(self.video_dir, f'environment_image_results_{self.timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results, output_dir 
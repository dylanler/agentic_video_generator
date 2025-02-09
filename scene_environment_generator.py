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
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_environment_prompts(self, environments):
        """Generate 10 prompts for each physical environment using Claude."""
        prompt = """
        For each physical environment description, create 10 different prompts that:
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
                    "prompt_number": integer,
                    "prompt_text": "detailed prompt with camera angle and character placement"
                }
            ]
        }
        """
        
        all_prompts = []
        for idx, env in enumerate(environments):
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.7,
                system="You are an expert at creating detailed image generation prompts.",
                messages=[{
                    "role": "user",
                    "content": f"Environment description: {env['scene_physical_environment']}\n\n{prompt}"
                }]
            )
            
            try:
                env_prompts = json.loads(response.content[0].text)
                all_prompts.append(env_prompts)
            except Exception as e:
                print(f"Error parsing prompts for environment {idx}: {str(e)}")
                continue
        
        # Save prompts to file
        output_path = f"scene_physical_environment_prompts_{self.timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(all_prompts, f, indent=2)
            
        return all_prompts, output_path
    
    def generate_environment_images(self, prompts_data, output_dir="scene_environment_images"):
        """Generate images for all prompts using multiprocessing."""
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
        
        return results, output_dir 
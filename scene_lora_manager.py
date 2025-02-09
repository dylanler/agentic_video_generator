import os
import json
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from fal_train_lora import LoraTrainer
from fal_lora_inference import FalLoraInference

def train_single_lora(args):
    env_idx, zip_path, trainer, timestamp = args
    result = None  # Initialize result to None
    try:
        trigger_word = f"ENV_{env_idx}_{timestamp}"
        print(f"Training LoRA for environment {env_idx} with trigger word: {trigger_word}")
        result = trainer.train_lora(zip_path, trigger_word)
        print(f"Raw training result for environment {env_idx}: {result}")
        
        # Save raw training result
        result_dir = os.path.join(os.path.dirname(zip_path), f"ENV_{env_idx}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"ENV_{env_idx}_{timestamp}_output.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved raw training result to: {result_path}")
        
        # Extract lora_path from the training result
        if isinstance(result, dict) and 'diffusers_lora_file' in result:
            lora_path = result['diffusers_lora_file']['url']
            print(f"Successfully extracted LoRA path from diffusers_lora_file: {lora_path}")
        else:
            raise ValueError(f"No diffusers_lora_file found in training result: {result}")
            
        print(f"Successfully extracted LoRA path for environment {env_idx}: {lora_path}")
        return {
            "environment_index": env_idx,
            "trigger_word": trigger_word,
            "lora_path": lora_path,
            "training_result": result,
            "result_path": result_path  # Include the path to the raw result file
        }
    except Exception as e:
        print(f"Error training LoRA for environment {env_idx}: {str(e)}")
        if result is not None:  # Only print result if it exists
            print(f"Full training result: {result}")  # Print full result for debugging
            # Try to save the failed result as well
            try:
                result_dir = os.path.join(os.path.dirname(zip_path), f"ENV_{env_idx}_{timestamp}_failed")
                os.makedirs(result_dir, exist_ok=True)
                result_path = os.path.join(result_dir, f"ENV_{env_idx}_{timestamp}_failed_output.json")
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved failed training result to: {result_path}")
            except Exception as save_error:
                print(f"Could not save failed training result: {str(save_error)}")
        return None

def generate_frame_pair(args):
    scene, lora_data, inference, output_dir = args
    try:
        print(f"Generating frames for scene {scene['scene_number']}")
        print(f"Using LoRA data: {lora_data}")
        
        scene_dir = os.path.join(output_dir, f"scene_{scene['scene_number']}")
        os.makedirs(scene_dir, exist_ok=True)
        
        # Construct base prompt with trigger word
        trigger_word = lora_data['trigger_word']
        base_prompt = f"{trigger_word}, high quality, masterpiece, best quality, "
        
        # Generate first frame
        first_frame_path = os.path.join(scene_dir, f"first_frame.jpg")
        print(f"Generating first frame: {first_frame_path}")
        first_frame_prompt = base_prompt + scene.get('first_frame_prompt', scene['scene_physical_environment'])
        print(f"First frame prompt: {first_frame_prompt}")
        first_frame_result = inference.run_inference(
            prompt=first_frame_prompt,
            lora_path=lora_data["lora_path"],
            output_path=first_frame_path
        )
        
        # Generate last frame
        last_frame_path = os.path.join(scene_dir, f"last_frame.jpg")
        print(f"Generating last frame: {last_frame_path}")
        last_frame_prompt = base_prompt + scene.get('last_frame_prompt', scene['scene_physical_environment'])
        print(f"Last frame prompt: {last_frame_prompt}")
        last_frame_result = inference.run_inference(
            prompt=last_frame_prompt,
            lora_path=lora_data["lora_path"],
            output_path=last_frame_path
        )
        
        # Verify files were created
        if not os.path.exists(first_frame_path):
            raise RuntimeError(f"First frame was not generated: {first_frame_path}")
        if not os.path.exists(last_frame_path):
            raise RuntimeError(f"Last frame was not generated: {last_frame_path}")
            
        print(f"Successfully generated frames for scene {scene['scene_number']}")
        
        return {
            "scene_number": scene["scene_number"],
            "first_frame_path": first_frame_path,
            "last_frame_path": last_frame_path,
            "first_frame_result": first_frame_result,
            "last_frame_result": last_frame_result,
            "first_frame_prompt": first_frame_prompt,
            "last_frame_prompt": last_frame_prompt
        }
    except Exception as e:
        print(f"Error generating frames for scene {scene['scene_number']}: {str(e)}")
        print(f"Scene data: {scene}")
        print(f"LoRA data: {lora_data}")
        return None

class SceneLoraManager:
    def __init__(self, video_dir):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trainer = LoraTrainer(os.getenv("FAL_API_KEY"))
        self.inference = FalLoraInference()
        self.video_dir = video_dir
    
    def load_existing_lora_results(self, trained_lora_dir):
        """Load existing LoRA training results from a directory."""
        lora_results_file = next(
            (f for f in os.listdir(trained_lora_dir) if f.startswith('lora_training_results_')),
            None
        )
        if not lora_results_file:
            raise ValueError(f"No lora_training_results file found in {trained_lora_dir}")
        
        with open(os.path.join(trained_lora_dir, lora_results_file), 'r') as f:
            return json.load(f)
    
    def prepare_training_data(self, image_results):
        """Prepare training data by organizing images into zip files per environment."""
        output_dir = os.path.join(self.video_dir, "lora_training_data")
        os.makedirs(output_dir, exist_ok=True)
        
        # Group images by environment
        env_images = {}
        for result in image_results:
            env_idx = result["environment_index"]
            if env_idx not in env_images:
                env_images[env_idx] = []
            env_images[env_idx].append(result["image_path"])
        
        # Create zip files for each environment
        zip_files = {}
        for env_idx, images in env_images.items():
            env_dir = os.path.join(output_dir, f"environment_{env_idx}")
            os.makedirs(env_dir, exist_ok=True)
            
            # Copy images to environment directory
            for img_path in images:
                shutil.copy2(img_path, env_dir)
            
            # Create zip file in the video directory
            zip_name = f"environment_{env_idx}_{self.timestamp}"
            zip_path = os.path.join(self.video_dir, f"{zip_name}.zip")
            shutil.make_archive(os.path.join(self.video_dir, zip_name), 'zip', env_dir)
            zip_files[env_idx] = zip_path
            
            # Save list of files in zip
            zip_contents = os.listdir(env_dir)
            zip_contents_path = os.path.join(self.video_dir, f"{zip_name}_contents.json")
            with open(zip_contents_path, 'w') as f:
                json.dump({"files": zip_contents}, f, indent=2)
        
        # Save zip files mapping
        zip_files_path = os.path.join(self.video_dir, f'lora_training_zips_{self.timestamp}.json')
        with open(zip_files_path, 'w') as f:
            json.dump(zip_files, f, indent=2)
        
        return zip_files
    
    def train_environment_loras(self, zip_files, environments):
        """Train LoRA models for each environment in parallel."""
        # Train LoRAs in parallel
        results = []
        with ProcessPoolExecutor() as executor:
            training_args = [(env_idx, zip_path, self.trainer, self.timestamp) 
                           for env_idx, zip_path in zip_files.items()]
            for result in executor.map(train_single_lora, training_args):
                if result:
                    results.append(result)
        
        # Save training results
        output_path = os.path.join(self.video_dir, f'lora_training_results_{self.timestamp}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results, output_path
    
    def generate_scene_frames(self, scenes, lora_results, output_dir=None):
        """Generate first and last frames for each scene using provided LoRA results."""
        if output_dir is None:
            output_dir = os.path.join(self.video_dir, "scene_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a mapping of environment descriptions to their indices
        env_mapping = {}
        for scene in scenes:
            env_desc = scene['scene_physical_environment']
            if env_desc not in env_mapping:
                env_mapping[env_desc] = len(env_mapping) + 1  # 1-based indexing
        
        print("Environment mapping:", env_mapping)
        print("Available LoRA results:", lora_results)
        
        # Prepare generation arguments
        generation_args = []
        for scene in scenes:
            # Validate scene duration
            if scene["scene_duration"] not in [5, 9]:
                raise ValueError(f"Invalid scene duration for scene {scene['scene_number']}: {scene['scene_duration']}. Must be either 5 or 9 seconds.")
            
            # Find matching LoRA for this scene's environment
            env_idx = env_mapping[scene['scene_physical_environment']]
            lora_data = next(
                (lr for lr in lora_results if lr["environment_index"] == env_idx),
                None
            )
            
            if not lora_data:
                print(f"Warning: No LoRA found for scene {scene['scene_number']}, environment {env_idx}")
                print(f"Environment description: {scene['scene_physical_environment']}")
                print(f"Available LoRAs: {[lr['environment_index'] for lr in lora_results]}")
                continue
            
            generation_args.append((scene, lora_data, self.inference, output_dir))
        
        # Generate frames in parallel
        results = []
        with ProcessPoolExecutor() as executor:
            for result in executor.map(generate_frame_pair, generation_args):
                if result:
                    results.append(result)
        
        if not results:
            raise RuntimeError("No frames were successfully generated")
        
        # Save frame generation results
        output_path = os.path.join(self.video_dir, f'frame_generation_results_{self.timestamp}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save frame paths mapping
        frame_paths = {
            result["scene_number"]: {
                "first_frame": result["first_frame_path"],
                "last_frame": result["last_frame_path"]
            }
            for result in results
        }
        frame_paths_file = os.path.join(self.video_dir, f'frame_paths_{self.timestamp}.json')
        with open(frame_paths_file, 'w') as f:
            json.dump(frame_paths, f, indent=2)
        
        return results, output_path 
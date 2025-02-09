import os
import json
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from fal_train_lora import LoraTrainer
from fal_lora_inference import FalLoraInference

def train_single_lora(args):
    env_idx, zip_path, trainer, timestamp = args
    try:
        trigger_word = f"ENV_{env_idx}_{timestamp}"
        result = trainer.train_lora(zip_path, trigger_word)
        return {
            "environment_index": env_idx,
            "trigger_word": trigger_word,
            "lora_path": result["lora_path"],
            "training_result": result
        }
    except Exception as e:
        print(f"Error training LoRA for environment {env_idx}: {str(e)}")
        return None

def generate_frame_pair(args):
    scene, video_idx, lora_data, inference, output_dir = args
    try:
        scene_dir = os.path.join(output_dir, f"scene_{scene['scene_number']}")
        os.makedirs(scene_dir, exist_ok=True)
        
        # Generate first frame
        first_frame_path = os.path.join(scene_dir, f"video_{video_idx}_first_frame.jpg")
        first_frame_result = inference.run_inference(
            prompt=f"{scene['first_frame_prompt']} {lora_data['trigger_word']}",
            lora_path=lora_data["lora_path"],
            output_path=first_frame_path
        )
        
        # Generate last frame
        last_frame_path = os.path.join(scene_dir, f"video_{video_idx}_last_frame.jpg")
        last_frame_result = inference.run_inference(
            prompt=f"{scene['last_frame_prompt']} {lora_data['trigger_word']}",
            lora_path=lora_data["lora_path"],
            output_path=last_frame_path
        )
        
        return {
            "scene_number": scene["scene_number"],
            "video_index": video_idx,
            "first_frame_path": first_frame_path,
            "last_frame_path": last_frame_path,
            "first_frame_result": first_frame_result,
            "last_frame_result": last_frame_result
        }
    except Exception as e:
        print(f"Error generating frames for scene {scene['scene_number']}, video {video_idx}: {str(e)}")
        return None

class SceneLoraManager:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trainer = LoraTrainer(os.getenv("FAL_API_KEY"))
        self.inference = FalLoraInference()
        
    def prepare_training_data(self, image_results, output_dir="lora_training_data"):
        """Prepare training data by organizing images into zip files per environment."""
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
            
            # Create zip file
            zip_path = f"{env_dir}.zip"
            shutil.make_archive(env_dir, 'zip', env_dir)
            zip_files[env_idx] = zip_path
        
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
        output_path = f"lora_training_results_{self.timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results, output_path
    
    def generate_scene_frames(self, scenes, lora_results, output_dir="scene_frames"):
        """Generate first and last frames for each video in each scene."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare generation arguments
        generation_args = []
        for scene in scenes:
            # Find matching LoRA for this scene's environment
            lora_data = next(
                (lr for lr in lora_results if lr["environment_index"] == scene["environment_index"]),
                None
            )
            if not lora_data:
                continue
                
            # Calculate number of videos needed based on scene duration
            if scene["scene_duration"] <= 9:
                num_videos = 1
            elif scene["scene_duration"] <= 14:
                num_videos = 2
            else:
                num_videos = 2
                
            for video_idx in range(num_videos):
                generation_args.append((scene, video_idx, lora_data, self.inference, output_dir))
        
        # Generate frames in parallel
        results = []
        with ProcessPoolExecutor() as executor:
            for result in executor.map(generate_frame_pair, generation_args):
                if result:
                    results.append(result)
        
        # Save frame generation results
        output_path = f"frame_generation_results_{self.timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results, output_path 
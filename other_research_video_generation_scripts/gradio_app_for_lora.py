import gradio as gr
import os
import json
from datetime import datetime
from video_generation4 import (
    generate_scene_metadata,
    calculate_total_duration,
    generate_narration_text,
    generate_narration_audio,
    generate_scenes,
    stitch_videos,
    SceneEnvironmentGenerator,
    SceneLoraManager,
    LUMA_VIDEO_GENERATION_DURATION_OPTIONS
)

# Example JSON formats
EXAMPLE_SCENE_METADATA = '''[
    {
        "scene_number": 1,
        "scene_name": "Opening Scene",
        "scene_physical_environment": "A sunlit forest clearing with tall pine trees...",
        "environment_index": 0,
        "scene_movement_description": "A young woman walks through the clearing...",
        "scene_emotions": "Peaceful, serene, with a hint of mystery",
        "scene_camera_movement": "Slow dolly shot following the character",
        "scene_duration": 9,
        "sound_effects_prompt": "Gentle breeze through trees, distant bird calls..."
    }
]'''

EXAMPLE_ENVIRONMENT_PROMPTS = '''[
    {
        "scene_physical_environment": "A sunlit forest clearing with tall pine trees surrounding it. Golden sunlight filters through the canopy, creating dappled patterns on the forest floor covered in pine needles. The air is crisp and clear, with a slight morning mist lingering between the trees."
    },
    {
        "scene_physical_environment": "A cozy cabin interior with warm wooden walls and a crackling fireplace. Soft, warm light from the fire illuminates the rustic furniture and creates dancing shadows on the walls. A large window shows the forest outside."
    }
]'''

def load_json_or_none(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def update_scene_metadata(scenes_json):
    try:
        scenes = json.loads(scenes_json)
        # Validate required fields including environment_index
        required_fields = ['scene_number', 'scene_name', 'scene_physical_environment', 
                         'environment_index', 'scene_movement_description', 'scene_emotions',
                         'scene_camera_movement', 'sound_effects_prompt']
        
        if isinstance(scenes, list):
            for scene in scenes:
                missing_fields = [field for field in required_fields if field not in scene]
                if missing_fields:
                    return None, f"Missing required fields in scene: {', '.join(missing_fields)}"
                if not isinstance(scene['environment_index'], int):
                    return None, f"environment_index must be an integer in scene {scene['scene_number']}"
        else:
            return None, "Scene metadata must be a list of scenes"
            
        return scenes, "Successfully updated scene metadata"
    except Exception as e:
        return None, f"Error parsing JSON: {str(e)}"

def update_environment_prompts(prompts_json):
    try:
        prompts = json.loads(prompts_json)
        # Validate environment prompts structure
        if not isinstance(prompts, list):
            return None, "Environment prompts must be a list"
            
        for i, prompt in enumerate(prompts):
            if 'scene_physical_environment' not in prompt:
                return None, f"Missing scene_physical_environment in environment {i}"
            
        return prompts, "Successfully updated environment prompts"
    except Exception as e:
        return None, f"Error parsing JSON: {str(e)}"

def process_video_generation(
    script_text,
    model_choice,
    metadata_only,
    num_scenes,
    temperature,
    scene_metadata_json,
    environment_prompts_json,
    narration_speed_factor,
    sound_effect_volume,
    narration_volume,
    progress=gr.Progress()
):
    try:
        # Create output directory based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = f"generated_videos/video_{timestamp}"
        os.makedirs(video_dir, exist_ok=True)
        
        # Update scene metadata if provided
        if scene_metadata_json:
            scenes, msg = update_scene_metadata(scene_metadata_json)
            if not scenes:
                return {
                    "metadata_output": None,
                    "video_output": None,
                    "error_output": msg,
                    "scene_json_output": None,
                    "environment_json_output": None
                }
        else:
            progress(0.1, desc="Analyzing script and generating scene metadata...")
            scenes = generate_scene_metadata(
                script_text,
                model_choice,
                num_scenes if num_scenes > 0 else None,
                temperature
            )
        
        # Save scene metadata
        scene_metadata_path = os.path.join(video_dir, f'scene_metadata_{timestamp}.json')
        save_json(scenes, scene_metadata_path)
        
        if metadata_only:
            return {
                "metadata_output": f"Scene metadata JSON generated in: {video_dir}",
                "video_output": None,
                "error_output": None,
                "scene_json_output": json.dumps(scenes, indent=2),
                "environment_json_output": None
            }

        # Calculate total video duration
        total_duration = calculate_total_duration(scenes)
        
        progress(0.2, desc="Generating narration...")
        narration_text, narration_text_path = generate_narration_text(
            script_text, 
            total_duration * narration_speed_factor,
            model_choice
        )
        narration_audio_path = generate_narration_audio(
            narration_text,
            total_duration,
            volume=narration_volume,
            speed_factor=narration_speed_factor
        )
        
        progress(0.3, desc="Generating environment prompts and images...")
        env_generator = SceneEnvironmentGenerator(video_dir)
        
        # Update environment prompts if provided
        if environment_prompts_json:
            env_prompts, msg = update_environment_prompts(environment_prompts_json)
            if not env_prompts:
                return {
                    "metadata_output": None,
                    "video_output": None,
                    "error_output": msg,
                    "scene_json_output": json.dumps(scenes, indent=2),
                    "environment_json_output": None
                }
        else:
            env_prompts, prompts_path = env_generator.generate_environment_prompts(scenes)
        
        # Save environment prompts
        env_prompts_path = os.path.join(video_dir, f'environment_prompts_{timestamp}.json')
        save_json(env_prompts, env_prompts_path)
        
        image_results, images_dir = env_generator.generate_environment_images(env_prompts)
        
        progress(0.5, desc="Training environment LoRAs and generating frames...")
        lora_manager = SceneLoraManager(video_dir)
        zip_files = lora_manager.prepare_training_data(image_results)
        lora_results, lora_path = lora_manager.train_environment_loras(zip_files, scenes)
        frame_results, frames_path = lora_manager.generate_scene_frames(scenes, lora_results)
        
        progress(0.7, desc="Generating videos and sound effects...")
        video_files, sound_effect_files = generate_scenes(scenes, frame_results)
        
        progress(0.9, desc="Stitching videos together...")
        final_video = stitch_videos(
            video_files,
            sound_effect_files,
            narration_audio_path,
            sound_effect_volume=sound_effect_volume,
            narration_volume=narration_volume
        )
        
        return {
            "metadata_output": f"Scene metadata and intermediate files generated in: {video_dir}",
            "video_output": final_video,
            "error_output": None,
            "scene_json_output": json.dumps(scenes, indent=2),
            "environment_json_output": json.dumps(env_prompts, indent=2)
        }
        
    except Exception as e:
        return {
            "metadata_output": None,
            "video_output": None,
            "error_output": f"Error: {str(e)}",
            "scene_json_output": None,
            "environment_json_output": None
        }

# Create Gradio interface
with gr.Blocks(title="Video Generation App") as app:
    gr.Markdown("# AI Video Generation from Script")
    
    with gr.Tabs():
        # Main Generation Tab
        with gr.Tab("Main Generation"):
            with gr.Row():
                with gr.Column():
                    # Basic Input Components
                    script_input = gr.Textbox(
                        label="Movie Script",
                        placeholder="Enter your movie script here...",
                        lines=10
                    )
                    model_choice = gr.Radio(
                        choices=["gemini", "claude"],
                        value="gemini",
                        label="Select Model"
                    )
                    metadata_only = gr.Checkbox(
                        label="Generate Metadata Only",
                        value=False
                    )
                    num_scenes = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=0,
                        step=1,
                        label="Number of Scenes (0 for auto-detection)"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    submit_btn = gr.Button("Generate Video")
                
                with gr.Column():
                    # Basic Output Components
                    metadata_output = gr.Textbox(
                        label="Metadata Output",
                        lines=3
                    )
                    video_output = gr.Video(
                        label="Generated Video"
                    )
                    error_output = gr.Textbox(
                        label="Error Messages",
                        lines=3
                    )
        
        # Advanced Settings Tab
        with gr.Tab("Advanced Settings"):
            with gr.Row():
                with gr.Column():
                    # Scene Settings
                    gr.Markdown("### Scene Settings")
                    scene_metadata_json = gr.TextArea(
                        label="Scene Metadata JSON (Optional)",
                        placeholder="Paste custom scene metadata JSON here...",
                        lines=10,
                        value=EXAMPLE_SCENE_METADATA
                    )
                    environment_prompts_json = gr.TextArea(
                        label="Environment Prompts JSON (Optional)",
                        placeholder="Paste custom environment prompts JSON here...",
                        lines=10,
                        value=EXAMPLE_ENVIRONMENT_PROMPTS
                    )
                
                with gr.Column():
                    # Audio Settings
                    gr.Markdown("### Audio Settings")
                    narration_speed_factor = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Narration Speed Factor"
                    )
                    sound_effect_volume = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Sound Effect Volume"
                    )
                    narration_volume = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                        label="Narration Volume"
                    )
        
        # Generated JSON Tab
        with gr.Tab("Generated JSON"):
            scene_json_output = gr.JSON(
                label="Generated Scene Metadata"
            )
            environment_json_output = gr.JSON(
                label="Generated Environment Prompts"
            )
        
        # Examples Tab
        with gr.Tab("Examples"):
            gr.Markdown("### Example Scene Metadata JSON Format")
            gr.Code(
                value=EXAMPLE_SCENE_METADATA,
                language="json"
            )
            gr.Markdown("### Example Environment Prompts JSON Format")
            gr.Code(
                value=EXAMPLE_ENVIRONMENT_PROMPTS,
                language="json"
            )
            gr.Markdown("""
            ### Important Notes:
            1. Each scene must have an `environment_index` field pointing to its corresponding environment
            2. Environment indices are 0-based (0, 1, 2, etc.)
            3. Scene durations must be either 5 or 9 seconds
            4. All fields shown in the examples are required
            5. Environment prompts should be detailed and consistent across scenes
            """)
    
    # Connect components
    submit_btn.click(
        fn=process_video_generation,
        inputs=[
            script_input,
            model_choice,
            metadata_only,
            num_scenes,
            temperature,
            scene_metadata_json,
            environment_prompts_json,
            narration_speed_factor,
            sound_effect_volume,
            narration_volume
        ],
        outputs=[
            metadata_output,
            video_output,
            error_output,
            scene_json_output,
            environment_json_output
        ]
    )

if __name__ == "__main__":
    app.launch()
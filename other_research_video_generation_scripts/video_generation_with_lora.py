import os
import json
import time
from datetime import datetime
from google import genai
from lumaai import LumaAI
import requests
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip
import argparse
import ast
from scene_environment_generator import SceneEnvironmentGenerator
from scene_lora_manager import SceneLoraManager
from dotenv import load_dotenv
from generate_narration import generate_narration_for_video

# Load environment variables
load_dotenv()

# Initialize clients
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
luma_client = LumaAI(auth_token=os.getenv("LUMAAI_API_KEY"))

# Add video duration configuration
LUMA_VIDEO_GENERATION_DURATION_OPTIONS = [5, 9]  # Duration in seconds

# Get current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define video directory path (but don't create it yet)
video_dir = f"luma_generated_videos/luma_generation_{timestamp}"

def generate_physical_environments(num_scenes, script, model="gemini"):
    print(f"There are {num_scenes} scenes in the script.")
    prompt = f"""
    Create a JSON array of a bunch of detailed physical environment descriptions based on the movie script.
    Each environment should be detailed and include:
    - Setting details
    - Lighting conditions
    - Weather and atmospheric conditions
    - Time of day
    - Key objects and elements in the scene
    - Maximum number of physical environments is 5
    
    Some scenes will reuse the same physical environment. Across multiple scenes, the physical environment should maintain the same physical environment across two or more scenes.
    Focus on creating a cohesive visual narrative with the physical environment descriptions.
    
    Return: array of objects with format:
    {{
        "scene_physical_environment": "detailed string description"
    }}
    """
    
    try:
        if model == "gemini":
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=[script, prompt],
                config={
                    'response_mime_type': 'application/json',
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'response_schema': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'scene_number': {'type': 'integer'},
                                'scene_physical_environment': {'type': 'string'}
                            },
                            'required': ['scene_number', 'scene_physical_environment']
                        }
                    }
                }
            )
            environments = response.parsed
        
        elif model == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            
            system_prompt = """You are an expert at describing physical environments for video scenes."""

            claude_environments_format = """
            {
                "environments": [
                    {
                        "scene_physical_environment": "detailed string description"
                    },
                    ...
                ]
            }
            """
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": f"""
                {script}\n\n{prompt}
                Output JSON format like this:
                {claude_environments_format}
                """}]
            )
            
            try:
                response_text = response.content[0].text
                environments = ast.literal_eval(response_text)
                environments = environments["environments"]
                print(f"There are {len(environments)} scene physical environments in the script.")
            except Exception as e:
                print("Raw response content:", response_text)
                raise RuntimeError(f"Failed to parse Claude's response: {e}")
        
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        os.makedirs(video_dir, exist_ok=True)
        json_path = os.path.join(video_dir, f'scene_physical_environment_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(environments, f, indent=2)
        
        return environments, json_path
        
    except Exception as e:
        raise e

def generate_metadata_without_environment(num_scenes, script, model="gemini"):
    prompt = f"""
    Create a JSON array of {num_scenes} detailed scene descriptions based on the movie script. Each scene should include:
    1. A descriptive scene name that captures the essence of the moment
    2. Movement descriptions including:
       - Character movements and actions
       - Character appereances should be described in detail along with ethnicity, gender, age, clothing, style, and any other relevant details
       - Character appearances should be consistent with the previous scene's character appearances
       - Object interactions and dynamics
       - Flow of action
    3. Emotional components including:
       - Scene mood and atmosphere
       - Emotional undertones
       - Visual emotional cues
    4. Camera movement specifications including:
       - Shot types (wide, medium, close-up)
       - Camera angles
       - Movement patterns (pan, tilt, dolly, etc.)
       - Camera movement should be smooth and fluid and not jarring
       - If a scene does not require camera movement, enter camera movement as "static"
    5. Sound effects focusing on:
       - Environmental sounds
       - Action-related sounds
       - Ambient atmosphere
       - Musical mood suggestions
    6. Take into account the previous scene's movement description, emotions, camera movement, and sound effects prompt when creating the next scene's movement description, emotions, camera movement, and sound effects prompt.
    7. The first scene should have no previous scene movement description, emotions, camera movement, and sound effects prompt, enter string "none".
    8. Also select the duration of the scene from the options of {LUMA_VIDEO_GENERATION_DURATION_OPTIONS} in integer value.

    Focus on creating a cohesive visual narrative without any dialogue.
    
    Return: array of objects with format:
    {{
        "scene_number": integer,
        "scene_name": "string value",
        "previous_scene_movement_description": "string value",
        "scene_movement_description": "string value",
        "previous_scene_emotions": "string value",
        "scene_emotions": "string value",
        "previous_scene_camera_movement": "string value",
        "scene_camera_movement": "string value",
        "previous_scene_duration": "integer value", # 5, 9, 14, or 18
        "scene_duration": "integer value", # 5, 9, 14, or 18
        "previous_scene_sound_effects_prompt": "string value",
        "sound_effects_prompt": "string value"
    }}
    """
    
    try:
        if model == "gemini":
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=[script, prompt],
                config={
                    'response_mime_type': 'application/json',
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'response_schema': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'scene_number': {'type': 'integer'},
                                'scene_name': {'type': 'string'},
                                'scene_movement_description': {'type': 'string'},
                                'scene_emotions': {'type': 'string'},
                                'scene_camera_movement': {'type': 'string'},
                                'sound_effects_prompt': {'type': 'string'}
                            },
                            'required': ['scene_number', 'scene_name', 'scene_movement_description', 
                                       'scene_emotions', 'scene_camera_movement', 'sound_effects_prompt']
                        }
                    }
                }
            )
            metadata = response.parsed
        
        elif model == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            
            system_prompt = """You are an expert at creating detailed scene descriptions for videos. 
            You are also an expert at creating sound effects prompts for videos.
            You will output all number of scenes needed to tell the story effectively.
            """

            claude_metadata_no_env_format = """
            {
                "scenes": [
                    {
                        "scene_number": "integer value",
                        "scene_name": "string value",
                        "previous_scene_movement_description": "string value",
                        "scene_movement_description": "string value",
                        "previous_scene_emotions": "string value",
                        "scene_emotions": "string value",
                        "previous_scene_camera_movement": "string value",
                        "scene_camera_movement": "string value",
                        "previous_scene_duration": "integer value", # 5, 9, or 14
                        "scene_duration": "integer value", # 5, 9, or 14
                        "previous_scene_sound_effects_prompt": "string value",
                        "sound_effects_prompt": "string value",
                        "first_frame_prompt": "string value",
                        "last_frame_prompt": "string value"
                    },
                    ...
                ]
            }
            """
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": f"""
                {script}\n\n{prompt}
                Output JSON format like this:
                {claude_metadata_no_env_format}
                Output all scenes needed to tell the story effectively. No explanation is needed.
                """}]
            )
            
            try:
                response_text = response.content[0].text
                metadata = ast.literal_eval(response_text)
                metadata = metadata["scenes"]
            except Exception as e:
                print("Raw response content:", response_text)
                raise RuntimeError(f"Failed to parse Claude's response: {e}")
        
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        json_path = os.path.join(video_dir, f'scene_metadata_no_env_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata, json_path
        
    except Exception as e:
        raise e

def combine_metadata_with_environment(num_scenes, script, metadata_path, environments_path, model="gemini"):
    # Load both JSON files
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    with open(environments_path, 'r') as f:
        environments = json.load(f)
    
    prompt = f"""
    Given a list of {num_scenes} scene metadata and a list of physical environments, select the most appropriate physical environment 
    for each scene to ensure scene continuity in the physical environment of the video.
    
    Return: array of complete scene descriptions, each containing all metadata fields plus the selected physical environment.
    """
    
    try:
        if model == "gemini":
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=[script, json.dumps(metadata), json.dumps(environments), prompt],
                config={
                    'response_mime_type': 'application/json',
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'response_schema': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'scene_number': {'type': 'integer'},
                                'scene_name': {'type': 'string'},
                                'scene_physical_environment': {'type': 'string'},
                                'scene_movement_description': {'type': 'string'},
                                'scene_emotions': {'type': 'string'},
                                'scene_camera_movement': {'type': 'string'},
                                'scene_duration': {'type': 'integer'},
                                'sound_effects_prompt': {'type': 'string'},
                                'first_frame_prompt': {'type': 'string'},
                                'last_frame_prompt': {'type': 'string'}
                            },
                            'required': ['scene_number', 'scene_name', 'scene_physical_environment',
                                       'scene_movement_description', 'scene_emotions',
                                       'scene_camera_movement', 'sound_effects_prompt']
                        }
                    }
                }
            )
            final_metadata = response.parsed
        
        elif model == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            
            system_prompt = """You are an expert at combining scene metadata with appropriate physical environments.
            """

            claude_metadata_with_env_format = """
            {
                "scenes": [
                    {
                        "scene_number": "integer value",
                        "scene_name": "string value",
                        "scene_physical_environment": "string value",
                        "scene_movement_description": "string value",
                        "scene_emotions": "string value",
                        "scene_camera_movement": "string value",
                        "scene_duration": "integer value", # 5, 9, or 14
                        "sound_effects_prompt": "string value"
                    },
                    ...
                ]
            }
            """
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0.7,
                system=system_prompt,
                messages=[{
                    "role": "user", 
                    "content": f'''
                    Script:\n{script}\n\nMetadata:\n{json.dumps(metadata)}\n\nEnvironments:\n{json.dumps(environments)}\n\n{prompt}
                    Output in JSON format like this:
                    {claude_metadata_with_env_format}
                    '''
                }]
            )
            
            try:
                response_text = response.content[0].text
                final_metadata = ast.literal_eval(response_text)
                final_metadata = final_metadata["scenes"]

            except Exception as e:
                print("Raw response content:", response_text)
                raise RuntimeError(f"Failed to parse Claude's response: {e}")
        
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        json_path = os.path.join(video_dir, f'scenes_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(final_metadata, f, indent=2)
        
        return final_metadata
        
    except Exception as e:
        raise e

def generate_scene_metadata(script, model="gemini"):
    try:
        # First, determine optimal number of scenes
        prompt = f"""
        Analyze this movie script and determine the optimal number of scenes needed to tell the story effectively.
        Consider that:
        - Each scene is either {LUMA_VIDEO_GENERATION_DURATION_OPTIONS} seconds long
        - Scenes should maintain visual continuity
        - The story should flow naturally
        - Complex actions may need multiple scenes
        - The story should be told in a way that is engaging and interesting to watch
        - Maximum number of scenes is 8
        Return only a single integer representing the optimal number of scenes. No explanation is needed.
        """
        
        if model == "gemini":
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=[script, prompt],
                config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40
                }
            )
            num_scenes = int(response.text.strip())
            
        elif model == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.7,
                system="You are an expert at analyzing scripts and determining optimal scene counts.",
                messages=[{"role": "user", "content": f"{script}\n\n{prompt}"}]
            )
            num_scenes = int(response.content[0].text.strip())
        
        print(f"LLM determined optimal number of scenes: {num_scenes}")
        
        # Continue with existing scene generation logic using determined num_scenes
        environments, env_path = generate_physical_environments(num_scenes, script, model)
        metadata, metadata_path = generate_metadata_without_environment(num_scenes, script, model)
        final_metadata = combine_metadata_with_environment(num_scenes, script, metadata_path, env_path, model)
        
        return final_metadata
        
    except Exception as e:
        if os.path.exists(video_dir):
            try:
                os.rmdir(video_dir)
            except OSError:
                pass
        raise e

def generate_scenes(scenes, frame_generation_results):
    scene_video_files = []  # List of video files
    sound_effect_files = []  # List of sound effect files
    
    for i, scene in enumerate(scenes):
        print(f"Generating videos for Scene {scene['scene_number']}")
        scene_duration = scene['scene_duration']
        
        # Validate scene duration
        if scene_duration not in [5, 9]:
            raise ValueError(f"Invalid scene duration: {scene_duration}. Must be either 5 or 9 seconds.")
        
        # Create scene-specific directory
        scene_dir = f"{video_dir}/scene_{scene['scene_number']}_vid_{timestamp}"
        os.makedirs(scene_dir, exist_ok=True)
        
        # Generate sound effect
        sound_effect_path = f"{scene_dir}/scene_{scene['scene_number']}_sound.mp3"
        print(f"Generating sound effect for Scene {scene['scene_number']}")
        try:
            sound_effect_generator = elevenlabs_client.text_to_sound_effects.convert(
                text=scene['sound_effects_prompt'],
                duration_seconds=scene_duration,
                prompt_influence=0.5
            )
            
            with open(sound_effect_path, 'wb') as f:
                for chunk in sound_effect_generator:
                    if chunk is not None:
                        f.write(chunk)
            
            sound_effect_files.append(sound_effect_path)
            print(f"Sound effect saved to: {sound_effect_path}")
        except Exception as e:
            print(f"Failed to generate sound effect: {e}")
            sound_effect_files.append(None)
        
        # Construct video generation prompt
        video_prompt = f"""
        {scene['scene_physical_environment']}
        
        Movement and Action:
        {scene['scene_movement_description']}
        
        Emotional Atmosphere:
        {scene['scene_emotions']}
        
        Camera Instructions:
        {scene['scene_camera_movement']}
        """
        
        # Get frame data for this video
        frame_data = next(
            (fr for fr in frame_generation_results 
             if fr["scene_number"] == scene["scene_number"]),
            None
        )
        
        if not frame_data:
            print(f"Warning: No frame data found for scene {scene['scene_number']}")
            continue
            
        # Get image URLs from frame results
        first_frame_url = frame_data["first_frame_result"]["images"][0]["url"]
        last_frame_url = frame_data["last_frame_result"]["images"][0]["url"]
        
        # Generate single video for the scene
        video_path = f"{scene_dir}/scene_{scene['scene_number']}_{timestamp}.mp4"
        
        # Create video generation with LoRA-generated frames
        generation_params = {
            "prompt": video_prompt.strip(),
            # "model": "ray-2",
            # "resolution": "720p",
            #"duration": f"{scene_duration}s",
            "keyframes": {
                "frame0": {
                    "type": "image",
                    "url": first_frame_url
                },
                "frame1": {
                    "type": "image",
                    "url": last_frame_url
                }
            }
        }
        
        print("Generate video name: ", video_path)
        print("Generating video with prompt: ", video_prompt.strip())
        print("Video duration: ", scene_duration)
        print("First frame URL: ", first_frame_url)
        print("Last frame URL: ", last_frame_url)
        print()
        generation = luma_client.generations.create(**generation_params)
        
        # Wait for completion
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                while True:
                    generation = luma_client.generations.get(id=generation.id)
                    if generation.state == "completed":
                        break
                    elif generation.state == "failed":
                        print(f"Generation state: {generation.state}")
                        print(f"Generation ID: {generation.id}")
                        print(f"Full generation object: {generation}")
                        print(f"Failure reason: {generation.failure_reason}")
                        raise RuntimeError(f"Generation failed: {generation.failure_reason}")
                    elif generation.state == "error":
                        print(f"Generation error state: {generation}")
                        raise RuntimeError(f"Generation error: {generation}")
                    print(f"Generation status: {generation.state}...")
                    time.sleep(3)
                break  # If we get here, generation was successful
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} attempts. Last error: {str(e)}")
                    raise
                print(f"Attempt {retry_count} failed: {str(e)}. Retrying...")
                time.sleep(5)  # Wait before retrying
        
        # Download video
        try:
            print(f"Downloading video from: {generation.assets.video}")
            response = requests.get(generation.assets.video, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            
            with open(video_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise RuntimeError(f"Video file is empty or not created: {video_path}")
                
            print(f"Video downloaded successfully to: {video_path}")
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            raise
        
        # Copy the video to the main directory
        final_video_path = f"{video_dir}/scene_{scene['scene_number']}_{timestamp}.mp4"
        import shutil
        shutil.copy2(video_path, final_video_path)
        scene_video_files.append(final_video_path)
        time.sleep(2)
    
    return scene_video_files, sound_effect_files

def calculate_total_duration(scenes):
    """Calculate total duration of all scenes in seconds"""
    return sum(scene['scene_duration'] for scene in scenes)

def generate_narration_text(scenes, total_duration, model="gemini"):
    """
    Generate narration text based on the scene metadata and desired duration.
    The narration should be timed to roughly match the video duration.
    """
    # Create a comprehensive scene description from metadata
    scene_descriptions = []
    for scene in scenes:
        description = f"""
        {scene['scene_name']}:
        Environment: {scene['scene_physical_environment']}
        Action: {scene['scene_movement_description']}
        Emotional Atmosphere: {scene['scene_emotions']}
        Camera Movement: {scene['scene_camera_movement']}
        """
        scene_descriptions.append(description)
    
    combined_description = "\n\n".join(scene_descriptions)
    
    prompt = f"""
    Create a narration script based on the scene descriptions below. The narration should:
    1. Be timed to take approximately {total_duration} seconds when read at a normal pace
    2. Output should be {total_duration * 2} number of words
    3. Provide context and atmosphere that enhances the visual elements
    4. Focus on describing key events, emotions, and revelations
    5. Maintain a consistent tone that matches the story's mood
    6. Be written in present tense
    7. Use clear, engaging language suitable for voice-over
    8. Include natural pauses and breaks in the pacing
    9. Flow smoothly between scenes while maintaining continuity
    
    Return the narration text only, without any formatting or additional notes.
    """
    
    try:
        if model == "gemini":
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=[combined_description, prompt],
                config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40
                }
            )
            narration = response.text
            
        elif model == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0.7,
                system="You are an expert at writing engaging narration scripts.",
                messages=[{"role": "user", "content": f"{combined_description}\n\n{prompt}"}]
            )
            narration = response.content[0].text
            
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        # Save narration text
        narration_path = os.path.join(video_dir, f'narration_text_{timestamp}.txt')
        with open(narration_path, 'w') as f:
            f.write(narration)
        
        return narration, narration_path
        
    except Exception as e:
        raise e

def get_narration_audio_path():
    """Check if narration audio has been generated separately"""
    result_path = os.path.join(video_dir, "narration_result.json")
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            result = json.load(f)
        audio_path = result.get("narration_audio_path")
        if audio_path and os.path.exists(audio_path):
            return audio_path
    return None

def stitch_videos(video_files, sound_effect_files, narration_audio_path=None):
    final_clips = []
    
    for video_file, sound_file in zip(video_files, sound_effect_files):
        try:
            video_clip = VideoFileClip(video_file)
            
            if sound_file and os.path.exists(sound_file):
                try:
                    # Load sound effect
                    audio_clip = AudioFileClip(sound_file)
                    
                    # If audio is longer than video, trim it
                    if audio_clip.duration > video_clip.duration:
                        audio_clip = audio_clip.subclip(0, video_clip.duration)
                    # If audio is shorter than video, loop it or pad with silence
                    elif audio_clip.duration < video_clip.duration:
                        # For simplicity, we'll just use the shorter duration
                        video_clip = video_clip.subclip(0, audio_clip.duration)
                    
                    # Combine video with sound effect
                    video_clip = video_clip.set_audio(audio_clip)
                except Exception as e:
                    print(f"Warning: Failed to process audio for {sound_file}: {str(e)}")
                    # Continue with video without audio if audio processing fails
            
            final_clips.append(video_clip)
        except Exception as e:
            print(f"Warning: Failed to process video {video_file}: {str(e)}")
            continue
    
    if not final_clips:
        raise RuntimeError("No video clips were successfully processed")
    
    # Concatenate all clips
    final_clip = concatenate_videoclips(final_clips)
    
    # Add narration audio if provided
    if narration_audio_path and os.path.exists(narration_audio_path):
        try:
            narration_audio = AudioFileClip(narration_audio_path)
            # Combine original audio with narration
            final_audio = CompositeVideoClip([final_clip]).audio
            if final_audio is not None:
                combined_audio = CompositeVideoClip([
                    final_clip.set_audio(final_audio.volumex(0.7)),  # Reduce original volume
                    final_clip.set_audio(narration_audio.volumex(1.0))  # Keep narration at full volume
                ]).audio
                final_clip = final_clip.set_audio(combined_audio)
            else:
                final_clip = final_clip.set_audio(narration_audio)
        except Exception as e:
            print(f"Warning: Failed to add narration audio: {str(e)}")
    
    # Write final video
    output_path = f"{video_dir}/luma_final_video_{timestamp}.mp4"
    final_clip.write_videofile(output_path)
    
    # Close all clips
    for clip in final_clips:
        clip.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate a video based on script analysis')
    parser.add_argument('--model', type=str, choices=['gemini', 'claude'], default='gemini',
                       help='Model to use for scene generation (default: gemini)')
    parser.add_argument('--metadata_only', action='store_true',
                       help='Only generate scene metadata JSON without video generation')
    parser.add_argument('--trained_lora_dir', type=str,
                       help='Directory containing previously trained LoRAs to use for frame generation')
    parser.add_argument('--narration_only', action='store_true',
                       help='Only generate narration audio and exit')
    parser.add_argument('--skip_narration', action='store_true',
                       help='Skip narration generation and use existing audio')
    args = parser.parse_args()

    if args.trained_lora_dir:
        print(f"Using pre-trained LoRAs and metadata from: {args.trained_lora_dir}")
        
        # Load existing scene metadata
        scenes_file = next(
            (f for f in os.listdir(args.trained_lora_dir) if f.startswith('scenes_')),
            None
        )
        if not scenes_file:
            raise ValueError(f"No scenes metadata file found in {args.trained_lora_dir}")
        
        scenes_path = os.path.join(args.trained_lora_dir, scenes_file)
        print(f"Loading scenes metadata from: {scenes_path}")
        with open(scenes_path, 'r') as f:
            scenes = json.load(f)
            
        print(f"Loaded metadata for {len(scenes)} scenes")
        
        # Load existing LoRA results
        lora_results_file = next(
            (f for f in os.listdir(args.trained_lora_dir) if f.startswith('lora_training_results_')),
            None
        )
        if not lora_results_file:
            raise ValueError(f"No lora_training_results file found in {args.trained_lora_dir}")
        
        lora_results_path = os.path.join(args.trained_lora_dir, lora_results_file)
        print(f"Loading LoRA results from: {lora_results_path}")
        with open(lora_results_path, 'r') as f:
            lora_results = json.load(f)
        
        if not lora_results:
            raise ValueError("LoRA results file is empty")
            
        print(f"Loaded {len(lora_results)} LoRA results")
        for lr in lora_results:
            print(f"LoRA {lr['environment_index']}: {lr['trigger_word']} -> {lr['lora_path']}")
        
        # Initialize LoRA manager and generate frames
        print("Generating frames using existing LoRAs...")
        lora_manager = SceneLoraManager(video_dir)
        frame_results, frames_path = lora_manager.generate_scene_frames(scenes, lora_results)
        
        if not frame_results:
            raise RuntimeError("No frames were generated. Check the logs above for errors.")
    else:
        # Generate scene metadata with LLM-determined number of scenes
        print(f"Analyzing script and generating scene metadata using {args.model}...")
        with open('movie_script2.txt', 'r') as f:
            script = f.read()
        scenes = generate_scene_metadata(script, args.model)
        
        if args.metadata_only:
            print(f"Scene metadata JSON generated in: {video_dir}")
            return
            
        # Generate environment prompts and images
        print("Generating environment prompts and images...")
        env_generator = SceneEnvironmentGenerator(video_dir)
        env_prompts, prompts_path = env_generator.generate_environment_prompts(scenes)
        image_results, images_dir = env_generator.generate_environment_images(env_prompts)
        
        # Train LoRAs and generate frames
        print("Training environment LoRAs and generating frames...")
        lora_manager = SceneLoraManager(video_dir)
        zip_files = lora_manager.prepare_training_data(image_results)
        lora_results, lora_path = lora_manager.train_environment_loras(zip_files, scenes)
        frame_results, frames_path = lora_manager.generate_scene_frames(scenes, lora_results)

    # Calculate total video duration
    total_duration = calculate_total_duration(scenes)
    print(f"Total video duration: {total_duration} seconds")
    
    # Handle narration
    narration_audio_path = None
    if not args.skip_narration:
        # Generate narration text
        print("Generating narration text from scene metadata...")
        narration_text, narration_text_path = generate_narration_text(scenes, total_duration, args.model)
        print(f"Narration text saved to: {narration_text_path}")
        
        if args.narration_only:
            print("\nGenerating narration audio...")
            narration_audio_path = generate_narration_for_video(narration_text, total_duration, video_dir)
            if narration_audio_path:
                print(f"Narration audio generated successfully: {narration_audio_path}")
                # Save the path for future use
                result = {
                    "narration_audio_path": narration_audio_path,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                result_path = os.path.join(video_dir, "narration_result.json")
                with open(result_path, 'w') as f:
                    json.dump(result, f)
            else:
                print("Failed to generate narration audio")
            return
    else:
        # Check for existing narration audio
        print("Checking for existing narration audio...")
        narration_audio_path = get_narration_audio_path()
        if not narration_audio_path:
            print("No narration audio found. Please run the script with --narration_only first")
            return
    
    # Generate videos and sound effects
    print("Generating videos and sound effects...")
    video_files, sound_effect_files = generate_scenes(scenes, frame_results)
    
    # Stitch videos with sound effects and narration
    print("Stitching videos together with sound effects and narration...")
    final_video = stitch_videos(video_files, sound_effect_files, narration_audio_path)
    
    print(f"Final video saved to: {final_video}")

if __name__ == "__main__":
    main()

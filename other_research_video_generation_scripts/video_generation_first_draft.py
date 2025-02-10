import os
import json
import time
from datetime import datetime
from google import genai
from lumaai import LumaAI
import requests
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip
from img_bucket import GCPImageUploader
import cv2
from elevenlabs import ElevenLabs
import argparse
import ast
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize clients
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Add video duration configuration
VIDEO_DURATION_SECONDS = 9  # Duration in seconds
VIDEO_DURATION_STRING = f"{VIDEO_DURATION_SECONDS}s"  # Duration as string with 's' suffix

luma_client = LumaAI(auth_token=os.getenv("LUMAAI_API_KEY"))

# Get current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define video directory path (but don't create it yet)
video_dir = f"generated_videos/video_{timestamp}"

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
    
    Remember that each scene is only {VIDEO_DURATION_SECONDS} seconds long.
    So if a scene is longer than that, the scene should maintain the same physical environment across two or more scenes.
    Focus on creating a cohesive visual narrative.
    
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
       - Character appereances should be described in detail
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
    5. Sound effects focusing on:
       - Environmental sounds
       - Action-related sounds
       - Ambient atmosphere
       - Musical mood suggestions
    6. Take into account the previous scene's movement description, emotions, camera movement, and sound effects prompt when creating the next scene's movement description, emotions, camera movement, and sound effects prompt.
    7. The first scene should have no previous scene movement description, emotions, camera movement, and sound effects prompt, enter string "none".
    
    Remember that each scene is only {VIDEO_DURATION_SECONDS} seconds long.
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
                        "previous_scene_sound_effects_prompt": "string value",
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
                                'sound_effects_prompt': {'type': 'string'}
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
        - Each scene is {VIDEO_DURATION_SECONDS} seconds long
        - Scenes should maintain visual continuity
        - The story should flow naturally
        - Complex actions may need multiple scenes
        - The story should be told in a way that is engaging and interesting to watch
        - Maximum number of scenes is 20
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

def generate_videos(scenes):
    video_files = []
    sound_effect_files = []
    last_frame_path = None
    last_frame_url = None
    
    uploader = GCPImageUploader()
    
    for i, scene in enumerate(scenes):
        print(f"Generating video for Scene {scene['scene_number']}")
        video_path = f"{video_dir}/scene_{scene['scene_number']}.mp4"
        sound_effect_path = f"{video_dir}/scene_{scene['scene_number']}_sound.mp3"
        
        # Generate sound effect first
        print(f"Generating sound effect for Scene {scene['scene_number']}")
        try:
            sound_effect_generator = elevenlabs_client.text_to_sound_effects.convert(
                text=scene['sound_effects_prompt'],
                duration_seconds=VIDEO_DURATION_SECONDS,
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
        
        # Construct comprehensive video generation prompt
        video_prompt = f"""
        {scene['scene_physical_environment']}
        
        Movement and Action:
        {scene['scene_movement_description']}
        
        Emotional Atmosphere:
        {scene['scene_emotions']}
        
        Camera Instructions:
        {scene['scene_camera_movement']}
        """
        
        # Create video generation with previous video's last frame if available
        generation_params = {
            "prompt": video_prompt.strip(),
            "model": "ray-2",
            "resolution": "540p",
            "duration": VIDEO_DURATION_STRING
        }
        
        if i > 0 and last_frame_url:
            generation_params["keyframes"] = {
                "frame0": {
                    "type": "image",
                    "url": last_frame_url
                }
            }
        
        generation = luma_client.generations.create(**generation_params)
        
        # Wait for completion
        while True:
            generation = luma_client.generations.get(id=generation.id)
            if generation.state == "completed":
                break
            elif generation.state == "failed":
                raise RuntimeError(f"Generation failed: {generation.failure_reason}")
            print("Dreaming...")
            time.sleep(3)
        
        # Download video
        response = requests.get(generation.assets.video, stream=True)
        with open(video_path, 'wb') as file:
            file.write(response.content)
        
        # Extract last frame using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count-1)
        ret, frame = cap.read()
        
        frame_path = f"{video_dir}/scene_{scene['scene_number']}_last_frame.jpg"
        if ret:
            cv2.imwrite(frame_path, frame)
            print(f"Successfully extracted last frame to: {frame_path}")
        else:
            raise RuntimeError(f"Failed to extract last frame from video: {video_path}")
        
        cap.release()
        
        # Upload frame to GCP and get signed URL for next scene
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            new_frame_url = uploader.upload_image(frame_path)
            if new_frame_url != last_frame_url:
                last_frame_url = new_frame_url
                print(f"Successfully uploaded frame with unique URL: {last_frame_url}")
                break
            print(f"Got duplicate URL, retrying... (attempt {retry_count + 1}/{max_retries})")
            time.sleep(2)
            retry_count += 1
        
        if retry_count == max_retries:
            raise RuntimeError(f"Failed to get unique frame URL for scene {scene['scene_number']}")
            
        video_files.append(video_path)
        time.sleep(2)
    
    return video_files, sound_effect_files

def stitch_videos(video_files, sound_effect_files):
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
    
    final_clip = concatenate_videoclips(final_clips)
    
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
    args = parser.parse_args()

    # Generate scene metadata with LLM-determined number of scenes
    print(f"Analyzing script and generating scene metadata using {args.model}...")
    with open('movie_script2.txt', 'r') as f:
        script = f.read()
    scenes = generate_scene_metadata(script, args.model)
    
    if args.metadata_only:
        print(f"Scene metadata JSON generated in: {video_dir}")
        return

    # Generate videos and sound effects
    print("Generating videos and sound effects...")
    video_files, sound_effect_files = generate_videos(scenes)
    
    # Stitch videos with sound effects
    print("Stitching videos together with sound effects...")
    final_video = stitch_videos(video_files, sound_effect_files)
    
    print(f"Final video saved to: {final_video}")

if __name__ == "__main__":
    main()

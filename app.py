import os
from datetime import datetime
import gradio as gr
from PIL import Image
from edit_picture import ImageProcessor
from flux import FluxImageGenerator
from runway import FalVideoGenerator
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip, vfx
import concurrent.futures
from functools import partial
import random
from suno import SongGenerator
import json
from fal_train_lora import LoraTrainer
from fal_lora_inference import FalLoraInference
import numpy as np
import zipfile
import string
import anthropic
from anthropic import Anthropic
import logging
import ast

# Initialize generators and uploader
flux_generator = FluxImageGenerator()
fal_generator = FalVideoGenerator()  # Replace luma_generator with fal_generator

# Initialize song generator
song_generator = SongGenerator()

# Create directories if they don't exist
def create_directories():
    directories = ['temp_uploaded', 'overlaid_img', 'background_img', 'gen_video', 'processed_img']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Get timestamped filename
def get_timestamped_filename(prefix, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

# New function to get a unique filename
def get_unique_filename(directory, prefix, extension):
    base_filename = get_timestamped_filename(prefix, extension)
    full_path = os.path.join(directory, base_filename)
    counter = 1
    while os.path.exists(full_path):
        new_filename = f"{prefix}_{counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"
        full_path = os.path.join(directory, new_filename)
        counter += 1
    return full_path

def generate_videos_parallel(*lora_images):
    valid_images = [img for img in lora_images if img is not None]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'gen_video_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    def generate_single_video(img, index):
        # Save numpy array as image file
        if isinstance(img, np.ndarray):
            img_path = os.path.join(output_dir, f"input_image_{index}.png")
            Image.fromarray(img).save(img_path)
        else:
            img_path = img

        # Generate a random hash
        random_hash = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        output_path = os.path.join(output_dir, f"output_video_{random_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        prompt = '''
        Create an epic commercial video for a product based on this image. 
        Include dynamic camera movements, dramatic lighting, and a sense of grandeur to showcase 
        the product's features and benefits.
        '''
        fal_generator.generate_video(prompt, img_path, output_path)
        return output_path

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_single_video, img, i) for i, img in enumerate(valid_images)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Pad with None values to always return 5 items
    return results + [None] * (5 - len(results))

def stitch_videos_with_audio(video_paths, audio_path):
    # Filter out None values from video_paths
    valid_video_paths = [path for path in video_paths if path is not None]
    
    if not valid_video_paths:
        print("No valid video paths provided.")
        return None
    
    try:
        video_clips = [VideoFileClip(path) for path in valid_video_paths]

        final_clip = concatenate_videoclips(video_clips)
        
        if audio_path:
            audio_clip = AudioFileClip(audio_path)
            # Trim audio to match video duration if necessary
            if audio_clip.duration > final_clip.duration:
                audio_clip = audio_clip.subclip(0, final_clip.duration)
            final_clip = final_clip.set_audio(audio_clip)
        
        # Ensure the output directory exists
        output_dir = 'gen_video'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = get_unique_filename(output_dir, 'final_stitched_video', 'mp4')
        final_clip.write_videofile(output_path)
        
        # Close all clips to free up resources
        for clip in video_clips:
            clip.close()
        if audio_path:
            audio_clip.close()
        final_clip.close()
        
        return output_path
    except Exception as e:
        print(f"Error in stitching videos: {str(e)}")
        return None

def generate_song(prompt):
    try:
        print(f"Generating song with prompt: {prompt}")
        mp3_files = song_generator.generate_song(prompt)
        if mp3_files:
            return mp3_files[0]  # Return the first generated song
    except Exception as e:
        print(f"Error generating song: {str(e)}")
    return None

# New function to train LoRA
def train_lora(zip_file, trigger_word):
    if zip_file is None:
        return "Please upload a zip file.", None
    
    if not zip_file.name:
        return "Invalid zip file name.", None
    
    # Use a default output directory
    output_dir = "lora_trained"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        trainer = LoraTrainer(os.getenv("FAL_API_KEY"))
        result = trainer.train_lora(zip_file.name, trigger_word)
        
        # Save the result to a file in the lora_trained directory
        output_file = os.path.join(output_dir, f"{trigger_word}_output.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        return f"LoRA training completed. Output saved to {output_file}", gr.Dropdown(choices=load_loras())
    except Exception as e:
        return f"Error during LoRA training: {str(e)}", None

# New function to load available LoRAs
def load_loras():
    loras = []
    lora_trained_dir = "lora_trained"
    if os.path.exists(lora_trained_dir):
        for file_name in os.listdir(lora_trained_dir):
            if file_name.endswith("_output.json"):
                file_path = os.path.join(lora_trained_dir, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    lora_url = data["diffusers_lora_file"]["url"]
                    trigger_word = file_name.replace("_output.json", "")
                    loras.append((trigger_word, lora_url))
    return loras

# New function to generate images using LoRA
def generate_lora_images(lora_url, prompt1, prompt2, prompt3, prompt4, prompt5):
    inference = FalLoraInference()
    prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]
    results = []
    
    # Create a new directory with timestamp for this batch of images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'lora_generated_images_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        if prompt:
            output_path = os.path.join(output_dir, f"lora_generated_image_{i+1}.jpg")
            result = inference.run_inference(prompt, lora_url, output_path)
            if result:
                results.append(output_path)
            else:
                results.append(None)
        else:
            results.append(None)
    
    return results

# Modified function to extract and load B-roll videos
def load_b_roll_videos(file_or_folder):
    if file_or_folder is None:
        return []
    
    b_roll_dir = 'b_roll_videos'
    os.makedirs(b_roll_dir, exist_ok=True)
    
    if isinstance(file_or_folder, list):
        # Handle multiple file uploads
        for file in file_or_folder:
            if zipfile.is_zipfile(file.name):
                with zipfile.ZipFile(file.name, 'r') as zip_ref:
                    zip_ref.extractall(b_roll_dir)
            elif file.name.lower().endswith(('.mp4', '.avi', '.mov')):
                dst_path = os.path.join(b_roll_dir, os.path.basename(file.name))
                with open(dst_path, 'wb') as dst_file:
                    dst_file.write(file.read())
    elif hasattr(file_or_folder, 'name') and zipfile.is_zipfile(file_or_folder.name):
        # Handle single zip file upload
        with zipfile.ZipFile(file_or_folder.name, 'r') as zip_ref:
            zip_ref.extractall(b_roll_dir)
    else:
        print("Invalid input: not a folder path, zip file path, or file object")
        return []
    
    b_roll_videos = []
    for file in os.listdir(b_roll_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov')):
            b_roll_videos.append(os.path.join(b_roll_dir, file))
    
    return b_roll_videos

# Update the update_b_roll_dropdown function
def update_b_roll_dropdown(file_or_folder):
    b_roll_videos = load_b_roll_videos(file_or_folder)
    return gr.Dropdown(choices=b_roll_videos, value=b_roll_videos[0] if b_roll_videos else None)

def generate_video_script(video_outputs, b_roll_metadata, audio_output, product_description):
    
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        return "Error: ANTHROPIC_API_KEY not found in .env file"

    client = Anthropic(api_key=anthropic_api_key)

    # Prepare the content for Claude
    content = "Create a video output script for a product commercial. Use the following information:\n\n"
    content += "Product shoot videos:\n"
    for i, video in enumerate(video_outputs):
        if video:
            content += f"- Video {i+1}: {os.path.abspath(video)}\n"

    content += "\nB-roll videos:\n"
    for video_name in os.listdir('b_roll_cut'):
        content += f"- {os.path.abspath(os.path.join('b_roll_cut', video_name))}\n"

    content += "\nB-roll videos and metadata:\n"
    for json_file in os.listdir(b_roll_metadata):
        if json_file.endswith('.json'):
            try:
                with open(os.path.join(b_roll_metadata, json_file), 'r') as f:
                    json_content = f.read()
                    metadata = json.loads(json_content)
                    content += f"- {os.path.abspath(os.path.join(b_roll_metadata, json_file))}:\n"
                    #content += f"  Prompt: {metadata.get('prompt', '')}\n"
                    content += f"  Video metadata: {json_content}\n\n"
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON file: {json_file}")
            except Exception as e:
                logging.error(f"Error processing file {json_file}: {str(e)}")

    content += "\nAudio file:\n"
    if audio_output:
        content += f"- {os.path.abspath(audio_output)}\n"

    content += f"\nProduct Description:\n{product_description}\n"

    json_format = '''
    {
        "video_sequence":
        [
            "/absolute/path/to/video1.mp4",
            "/absolute/path/to/video2.mp4",
            "/absolute/path/to/video3.mp4"
        ]
    }
    '''

    content += '''\nCreate a script that alternates between product shoot videos and b-roll videos. 
    Specify which videos to use and in what order. 
    The final output should be a list of video filenames to be stitched together with the audio. 
    Include a maximum of 5 b-roll videos in the sequence.
    Use the exact filenames provided with their absolute paths.
    Make sure the b-roll aesthetics inserted in the list are of similar style to the product shoot videos and align with the product description.
    Ensure the sequence tells a cohesive story about the product.\n
    '''
    content += f"Ensure that the output is a json response with the format {json_format}"
    content += "Start the output with {"

    print("Final content: ", content)

    # Get response from Claude using the Messages API
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": content}
        ]
    )

    print("Response from Claude: ", response.content[0].text)

    # Ensure that the response is json formatted correctly
    try:
        video_files = json.loads(response.content[0].text)
    except json.JSONDecodeError as e:
        print("Error decoding JSON response: ", str(e))
        video_files = video_outputs
    
    print("Parsed video files from script: ", video_files)
    
    return video_files

def resize_and_pad_video(video_path, target_size=(1080, 1920)):
    clip = VideoFileClip(video_path)
    
    # Calculate the aspect ratio of the target size
    target_aspect = target_size[0] / target_size[1]
    
    # Resize the video while maintaining its aspect ratio
    if clip.w / clip.h > target_aspect:
        new_height = int(clip.h * target_size[0] / clip.w)
        resized_clip = clip.resize(width=target_size[0])
    else:
        new_width = int(clip.w * target_size[1] / clip.h)
        resized_clip = clip.resize(height=target_size[1])
    
    # Create a black background
    background = ColorClip(size=target_size, color=(0,0,0))
    
    # Center the resized video on the background
    x_offset = max((target_size[0] - resized_clip.w) // 2, 0)
    y_offset = max((target_size[1] - resized_clip.h) // 2, 0)
    
    final_clip = CompositeVideoClip([background, resized_clip.set_position((x_offset, y_offset))])
    final_clip = final_clip.set_duration(clip.duration)
    
    return final_clip

def stitch_new_video(video_outputs, audio_output, product_description):
    b_roll_metadata = 'b_roll_metadata'

    print("Valid video outputs: ", video_outputs)
    
    valid_video_outputs = [v for v in video_outputs if v is not None]
    print("Filtered video outputs: ", valid_video_outputs)

    if not valid_video_outputs:
        print("No valid video outputs found.")
        return None

    video_file_json = generate_video_script(valid_video_outputs, b_roll_metadata, audio_output, product_description)
    video_files = video_file_json["video_sequence"]
    video_files = video_files[:random.randint(8, 12)]
    print("Final video files to be stitched: ", video_files)

    if not video_files:
        print("No video files found in the script. Using valid video outputs instead.")
        video_files = valid_video_outputs
    
    # Resize and pad all videos to 9:16 aspect ratio
    resized_clips = [resize_and_pad_video(video_path) for video_path in video_files]
    
    # Concatenate the resized clips
    final_clip = concatenate_videoclips(resized_clips)
    
    if audio_output:
        audio_clip = AudioFileClip(audio_output)
        if audio_clip.duration > final_clip.duration:
            audio_clip = audio_clip.subclip(0, final_clip.duration)
        final_clip = final_clip.set_audio(audio_clip)
    
    output_dir = 'gen_video'
    os.makedirs(output_dir, exist_ok=True)
    output_path = get_unique_filename(output_dir, 'final_stitched_video', 'mp4')
    
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Close all clips
    for clip in resized_clips:
        clip.close()
    if audio_output:
        audio_clip.close()
    final_clip.close()
    
    return output_path

with gr.Blocks() as demo:
    gr.Markdown("# Auto Product Commercial")

    gr.Markdown("## Train LoRA Model")
    with gr.Row():
        zip_file = gr.File(label="Upload Zip Folder of Images to Train LoRA")
        trigger_word = gr.Textbox(label="Enter trigger word")
    train_btn = gr.Button("Train LoRA")
    train_output = gr.Textbox(label="Training Output")

    gr.Markdown("## Product Description")
    product_description = gr.Textbox(label="Enter Product Description", lines=3)

    gr.Markdown("## Generate Images with LoRA")
    lora_dropdown = gr.Dropdown(label="Select LoRA", choices=load_loras())

    train_btn.click(
        train_lora,
        inputs=[zip_file, trigger_word],
        outputs=[train_output, lora_dropdown]
    )

    with gr.Row():
        lora_prompts = [gr.Textbox(label=f"Prompt {i+1}") for i in range(5)]
    generate_lora_btn = gr.Button("Generate LoRA Images")
    with gr.Row():
        lora_outputs = [gr.Image(label=f"LoRA Generated Image {i+1}") for i in range(5)]

    generate_lora_btn.click(
        generate_lora_images,
        inputs=[lora_dropdown] + lora_prompts,
        outputs=lora_outputs
    )
    
    video_btn = gr.Button("Generate Videos")
    with gr.Row():
        video_outputs = [gr.Video(label=f"Generated Video {i+1}") for i in range(5)]

    gr.Markdown("## Upload B-roll Videos")
    b_roll_input = gr.File(label="Upload B-roll Videos or Zip File", file_types=[".mp4", ".avi", ".mov", ".zip"], file_count="multiple")
    upload_b_roll_btn = gr.Button("Load B-roll Videos")
    with gr.Row():
        b_roll_dropdown = gr.Dropdown(label="Select B-roll Video", choices=[])
        b_roll_video = gr.Video(label="Selected B-roll Video")

    # Replace audio_input with text input for song prompt
    song_prompt = gr.Textbox(label="Enter song prompt")
    generate_song_btn = gr.Button("Generate Song")
    audio_output = gr.Audio(label="Generated Song", type="filepath")

    generate_song_btn.click(
        generate_song,
        inputs=[song_prompt],
        outputs=[audio_output]
    )
    
    video_btn.click(
        generate_videos_parallel,
        inputs=lora_outputs,
        outputs=video_outputs
    )
    
    stitch_btn = gr.Button("Stitch Video Together")
    final_video_output = gr.Video(label="Final Stitched Video")
    
    stitch_btn.click(
        lambda *args: stitch_videos_with_audio(args[:-1], args[-1]),
        inputs=video_outputs + [audio_output],
        outputs=final_video_output
    )

    stitch_new_btn = gr.Button("Stitch With B Rolls")
    final_new_video_output = gr.Video(label="New Stitched Video")



    stitch_new_btn.click(
        lambda *args: stitch_new_video(args[:-2], args[-2], args[-1]),
        inputs=video_outputs + [audio_output, product_description],
        outputs=final_new_video_output
    )

    upload_b_roll_btn.click(
        update_b_roll_dropdown,
        inputs=[b_roll_input],
        outputs=[b_roll_dropdown]
    )

    b_roll_dropdown.change(
        lambda x: x,
        inputs=[b_roll_dropdown],
        outputs=[b_roll_video]
    )

# make the app live
demo.launch()
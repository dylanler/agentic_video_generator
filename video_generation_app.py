import gradio as gr
import os
import json
from video_generation import generate_scene_metadata, generate_scenes, stitch_videos, calculate_total_duration, generate_narration_text, generate_narration_audio
from dotenv import load_dotenv
import tempfile
import shutil

def save_api_keys(gemini_key, eleven_labs_key, lumaai_key, anthropic_key, fal_key, bucket_name, credentials_file_obj):
    try:
        # Save the credentials file with original filename
        if credentials_file_obj is not None:
            original_filename = credentials_file_obj.name
            credentials_path = os.path.join(os.getcwd(), original_filename)
            with open(credentials_path, "wb") as f:
                f.write(credentials_file_obj)
        else:
            return "Error: GCP credentials file is required"
    
        # Create or update .env file
        env_content = f"""GEMINI_API_KEY={gemini_key}
ELEVEN_LABS_API_KEY={eleven_labs_key}
LUMAAI_API_KEY={lumaai_key}
ANTHROPIC_API_KEY={anthropic_key}
FAL_API_KEY={fal_key}
BUCKET_NAME={bucket_name}
CREDENTIALS_FILE={original_filename}
"""
        with open(".env", "w") as f:
            f.write(env_content)
        
        # Reload environment variables
        load_dotenv(override=True)
        return "API keys and credentials saved successfully!"
    except Exception as e:
        return f"Error saving credentials: {str(e)}"

def generate_video(script_text, model_choice="gemini", metadata_only=False):
    try:
        if not os.getenv("CREDENTIALS_FILE") or not os.path.exists(os.getenv("CREDENTIALS_FILE")):
            return "Error: GCP credentials file not found. Please set up your API keys first.", None
            
        # Generate scene metadata
        scenes = generate_scene_metadata(script_text, model_choice)
        
        if metadata_only:
            return json.dumps(scenes, indent=2), None
        
        # Calculate total duration
        total_duration = calculate_total_duration(scenes)
        
        # Generate narration
        narration_text, narration_text_path = generate_narration_text(scenes, total_duration, model_choice)
        narration_audio_path = generate_narration_audio(narration_text, total_duration)
        
        # Generate videos and sound effects
        video_files, sound_effect_files = generate_scenes(scenes)
        
        # Stitch videos with sound effects and narration
        final_video = stitch_videos(video_files, sound_effect_files, narration_audio_path)
        
        return json.dumps(scenes, indent=2), final_video
    except Exception as e:
        return str(e), None

# Create Gradio interface
with gr.Blocks(title="Video Generation System") as app:
    gr.Markdown("# Video Generation System")
    
    with gr.Tab("API Keys Setup"):
        gr.Markdown("""## Setup API Keys
        
Please provide your API keys and GCP credentials:
1. Get your API keys from respective services
2. Create a Google Cloud project and download your service account credentials JSON file
3. Enter all the required information below
""")
        with gr.Row():
            with gr.Column():
                gemini_key = gr.Textbox(label="Gemini API Key", type="password")
                eleven_labs_key = gr.Textbox(label="ElevenLabs API Key", type="password")
                lumaai_key = gr.Textbox(label="LumaAI API Key", type="password")
            with gr.Column():
                anthropic_key = gr.Textbox(label="Anthropic API Key", type="password")
                fal_key = gr.Textbox(label="FAL API Key", type="password")
                bucket_name = gr.Textbox(label="GCP Bucket Name")
        
        gr.Markdown("""### GCP Credentials
Upload your Google Cloud service account credentials JSON file. You can create one from the Google Cloud Console:
1. Go to IAM & Admin > Service Accounts
2. Create a new service account or select existing one
3. Create a new key (JSON type)
4. Upload the downloaded JSON file here
""")
        credentials_file = gr.File(label="GCP Credentials JSON File", file_types=[".json"])
        save_btn = gr.Button("Save API Keys")
        api_status = gr.Textbox(label="Status", interactive=False)
        
        save_btn.click(
            save_api_keys,
            inputs=[gemini_key, eleven_labs_key, lumaai_key, anthropic_key, fal_key, bucket_name, credentials_file],
            outputs=api_status
        )
    
    with gr.Tab("Video Generation"):
        gr.Markdown("## Generate Video from Script")
        with gr.Row():
            with gr.Column():
                script_input = gr.Textbox(label="Movie Script", lines=10, placeholder="Enter your movie script here...")
                model_choice = gr.Radio(choices=["gemini", "claude"], label="Model Choice", value="gemini")
                metadata_only = gr.Checkbox(label="Generate Metadata Only", value=False)
                generate_btn = gr.Button("Generate Video")
            with gr.Column():
                metadata_output = gr.Textbox(label="Generated Metadata", interactive=False)
                video_output = gr.Video(label="Generated Video")
        
        generate_btn.click(
            generate_video,
            inputs=[script_input, model_choice, metadata_only],
            outputs=[metadata_output, video_output]
        )

if __name__ == "__main__":
    app.launch() 
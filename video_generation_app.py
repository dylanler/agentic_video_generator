import gradio as gr
import os
import json
from video_generation import generate_scene_metadata, generate_scenes, stitch_videos, calculate_total_duration, generate_narration_text, generate_narration_audio
from dotenv import load_dotenv
import tempfile
import shutil
from datetime import datetime

def save_api_keys(gemini_key, eleven_labs_key, lumaai_key, anthropic_key, fal_key, bucket_name, credentials_file_obj):
    try:
        # Save the credentials file with original filename
        if credentials_file_obj is not None:
            # Get the temporary file path from Gradio's file object
            temp_path = credentials_file_obj.name
            original_filename = os.path.basename(temp_path)
            target_path = os.path.join(os.getcwd(), original_filename)
            
            # Copy the file from temp location to target location
            shutil.copy2(temp_path, target_path)
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

def load_custom_environments(file_obj):
    try:
        if file_obj is None:
            return None
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return json.loads(content)
    except Exception as e:
        print(f"Error loading custom environments: {str(e)}")
        return None

def generate_video(
    script_text, 
    model_choice="gemini",
    video_engine="luma",
    metadata_only=False,
    max_scenes=12,
    max_environments=3,
    custom_env_prompt=None,
    custom_environments_file=None,
    skip_narration=False,
    skip_sound_effects=False,
    initial_image_path=None,
    initial_image_prompt=None,
    first_frame_image_gen=False,
    image_gen_model="fal"
):
    try:
        if not os.getenv("CREDENTIALS_FILE") or not os.path.exists(os.getenv("CREDENTIALS_FILE")):
            return "Error: GCP credentials file not found. Please set up your API keys first.", None
        
        if initial_image_path and initial_image_prompt:
            return "Error: Cannot provide both initial image path and prompt. Please choose one.", None
            
        if (initial_image_prompt or first_frame_image_gen) and not (
            (image_gen_model == 'luma' and os.getenv("LUMAAI_API_KEY")) or 
            (image_gen_model == 'fal' and os.getenv("FAL_KEY"))
        ):
            return f"Error: {image_gen_model.upper()} API key is required for image generation.", None
        
        # Load custom environments if provided
        custom_environments = None
        if custom_environments_file:
            custom_environments = load_custom_environments(custom_environments_file)
            if custom_environments is None:
                return "Error: Invalid custom environments JSON file format", None
        
        # Generate scene metadata with custom parameters
        scenes = generate_scene_metadata(
            script_text, 
            model=model_choice,
            max_scenes=max_scenes,
            max_environments=max_environments,
            custom_env_prompt=custom_env_prompt,
            custom_environments=custom_environments,
            video_engine=video_engine
        )
        
        if metadata_only:
            return json.dumps(scenes, indent=2), None
        
        # Generate narration if not skipped
        narration_audio_path = None
        if not skip_narration:
            # Calculate total duration only if needed for narration
            total_duration = calculate_total_duration(scenes)
            narration_text, narration_text_path = generate_narration_text(scenes, total_duration, model_choice)
            narration_audio_path = generate_narration_audio(narration_text, total_duration)
        
        # Generate videos and sound effects
        print("Generating videos and sound effects...")
        video_files, sound_effect_files = generate_scenes(
            scenes, 
            video_engine, 
            skip_sound_effects,
            initial_image_path=initial_image_path.name if initial_image_path else None,
            initial_image_prompt=initial_image_prompt,
            first_frame_image_gen=first_frame_image_gen,
            image_gen_model=image_gen_model
        )
        
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
                script_input = gr.Textbox(
                    label="Movie Script", 
                    lines=10, 
                    placeholder="Enter your movie script here..."
                )
                
                with gr.Row():
                    use_random_script = gr.Checkbox(
                        label="Generate Random Script",
                        value=False,
                        info="Generate a random script using the selected model instead of using the script above"
                    )
                    random_script_btn = gr.Button("Preview Random Script")
                
                model_choice = gr.Radio(
                    choices=["gemini", "claude"], 
                    label="Model Choice", 
                    value="gemini"
                )
                video_engine = gr.Radio(
                    choices=["luma", "ltx"],
                    label="Video Engine",
                    value="luma",
                    info="Choose video generation engine. Note: LTX only supports 5-second videos, while Luma supports 5, 9, 14, or 18 seconds"
                )

                gr.Markdown("### Initial Frame Options (Optional)")
                gr.Markdown("Choose ONE of the following options to set the starting frame of the first video:")
                with gr.Row():
                    initial_image_path = gr.File(
                        label="Upload Initial Image",
                        file_types=["image"],
                        type="filepath"
                    )
                with gr.Row():
                    initial_image_prompt = gr.Textbox(
                        label="Initial Image Generation Prompt",
                        lines=5,
                        placeholder="Enter prompt to generate initial image..."
                    )
                    image_gen_model = gr.Radio(
                        choices=["luma", "fal"],
                        label="Image Generation Model",
                        value="fal",
                        info="Choose which model to use for image generation"
                    )
                with gr.Row():
                    first_frame_image_gen = gr.Checkbox(
                        label="Generate First Frame Images",
                        value=False,
                        info="Generate first frame images for each scene using the selected image model"
                    )
                metadata_only = gr.Checkbox(label="Generate Metadata Only", value=False)
                generate_btn = gr.Button("Generate Video")

                with gr.Row():
                    skip_narration = gr.Checkbox(
                        label="Skip Narration",
                        value=False,
                        info="Skip generating narration audio"
                    )
                    skip_sound_effects = gr.Checkbox(
                        label="Skip Sound Effects",
                        value=False,
                        info="Skip generating sound effects"
                    )
                max_scenes = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=12,
                    step=1,
                    label="Maximum Number of Scenes"
                )
                max_environments = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Maximum Number of Environments"
                )
                custom_env_prompt = gr.Textbox(
                    label="Custom Environment Description Prompt (Optional)",
                    lines=5,
                    placeholder="Enter custom prompt for generating physical environment descriptions..."
                )
                custom_environments_file = gr.File(
                    label="Custom Environment Descriptions JSON (Optional)",
                    file_types=[".json"],
                    type="binary"
                )
                
            with gr.Column():
                metadata_output = gr.Textbox(
                    label="Generated Metadata", 
                    interactive=False,
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
                video_output = gr.Video(label="Generated Video")
        
        # Add example JSON format help
        gr.Markdown("""
        ### Custom Environment JSON Format Example:
        ```json
        [
            {
                "scene_physical_environment": "A dimly lit urban alley at night, wet cobblestones reflecting neon signs..."
            },
            {
                "scene_physical_environment": "A sun-drenched beach at golden hour, gentle waves lapping at the shore..."
            }
        ]
        ```
        """)
        
        def generate_and_show_progress(
            script_input, 
            model_choice, 
            video_engine, 
            metadata_only, 
            max_scenes, 
            max_environments, 
            custom_env_prompt, 
            custom_environments_file, 
            skip_narration, 
            skip_sound_effects,
            initial_image_path,
            initial_image_prompt,
            first_frame_image_gen,
            image_gen_model,
            use_random_script
        ):
            if initial_image_path and initial_image_prompt:
                return {
                    metadata_output: "Error: Cannot provide both initial image path and prompt. Please choose one.",
                    video_output: None
                }
                
            if (initial_image_prompt or first_frame_image_gen) and not (
                (image_gen_model == 'luma' and os.getenv("LUMAAI_API_KEY")) or 
                (image_gen_model == 'fal' and os.getenv("FAL_KEY"))
            ):
                return {
                    metadata_output: f"Error: {image_gen_model.upper()} API key is required for image generation.",
                    video_output: None
                }
                
            # Generate random script if requested
            if use_random_script:
                try:
                    import random_script_generator
                    script_data = random_script_generator.generate_random_script(model_choice)
                    script = script_data["script"]
                    
                    # Display the generated script in the script input box
                    script_input = script
                    
                    # Create a message to show the user that a random script was generated
                    random_script_info = f"Using randomly generated script with model: {model_choice}\n\n"
                    random_script_info += f"Elements used:\n"
                    random_script_info += f"- Characters: {', '.join(script_data['elements']['characters'])}\n"
                    random_script_info += f"- Objects: {', '.join(script_data['elements']['objects'])}\n"
                    random_script_info += f"- Environment: {script_data['elements']['environment']}\n"
                    random_script_info += f"- Atmosphere: {script_data['elements']['atmosphere']}\n"
                    random_script_info += f"- Storyline: {script_data['elements']['storyline']}\n"
                    random_script_info += f"- Artistic Style: {script_data['elements']['artistic_style']}\n\n"
                    
                    # Save the script to a file in the current directory
                    script_file_path = f"random_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    elements_file_path = f"random_script_elements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    with open(script_file_path, "w") as f:
                        f.write(script)
                    
                    with open(elements_file_path, "w") as f:
                        json.dump(script_data["elements"], f, indent=2)
                    
                    print(f"Random script generated and saved to: {script_file_path}")
                    print(f"Script elements saved to: {elements_file_path}")
                except Exception as e:
                    return {
                        metadata_output: f"Error generating random script: {str(e)}",
                        video_output: None
                    }
            else:
                script = script_input
                random_script_info = ""
            
            # Generate video
            try:
                scenes_json, final_video = generate_video(
                    script, 
                    model_choice=model_choice,
                    video_engine=video_engine,
                    metadata_only=metadata_only,
                    max_scenes=max_scenes,
                    max_environments=max_environments,
                    custom_env_prompt=custom_env_prompt,
                    custom_environments_file=custom_environments_file,
                    skip_narration=skip_narration,
                    skip_sound_effects=skip_sound_effects,
                    initial_image_path=initial_image_path,
                    initial_image_prompt=initial_image_prompt,
                    first_frame_image_gen=first_frame_image_gen,
                    image_gen_model=image_gen_model
                )
                
                if random_script_info and not metadata_only:
                    scenes_json = random_script_info + scenes_json
                
                return {
                    metadata_output: scenes_json,
                    video_output: final_video
                }
            except Exception as e:
                return {
                    metadata_output: f"Error: {str(e)}",
                    video_output: None
                }
        
        def preview_random_script(model_choice):
            try:
                import random_script_generator
                script_data = random_script_generator.generate_random_script(model_choice)
                script = script_data["script"]
                
                # Create a message with the script and its elements
                preview_text = f"Randomly generated script with model: {model_choice}\n\n"
                preview_text += f"Elements used:\n"
                preview_text += f"- Characters: {', '.join(script_data['elements']['characters'])}\n"
                preview_text += f"- Objects: {', '.join(script_data['elements']['objects'])}\n"
                preview_text += f"- Environment: {script_data['elements']['environment']}\n"
                preview_text += f"- Atmosphere: {script_data['elements']['atmosphere']}\n"
                preview_text += f"- Storyline: {script_data['elements']['storyline']}\n"
                preview_text += f"- Artistic Style: {script_data['elements']['artistic_style']}\n\n"
                preview_text += f"Script:\n{'-' * 40}\n{script}\n{'-' * 40}"
                
                return script, preview_text
            except Exception as e:
                return "", f"Error generating random script: {str(e)}"
        
        # Connect the random script preview button
        random_script_btn.click(
            preview_random_script,
            inputs=[model_choice],
            outputs=[script_input, metadata_output]
        )
        
        # Connect the generate button
        generate_btn.click(
            generate_and_show_progress,
            inputs=[
                script_input, 
                model_choice, 
                video_engine, 
                metadata_only, 
                max_scenes, 
                max_environments, 
                custom_env_prompt, 
                custom_environments_file, 
                skip_narration, 
                skip_sound_effects,
                initial_image_path,
                initial_image_prompt,
                first_frame_image_gen,
                image_gen_model,
                use_random_script
            ],
            outputs=[metadata_output, video_output]
        )

if __name__ == "__main__":
    app.launch() 
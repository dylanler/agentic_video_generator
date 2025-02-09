import os
from datetime import datetime
from elevenlabs import ElevenLabs
from moviepy.editor import AudioFileClip
from eleven_labs_tts import generate_speech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_narration_for_video(narration_text, target_duration, output_dir):
    """
    Standalone function to generate narration audio for a video.
    This should be run separately from the main video generation to avoid API waste.
    
    Args:
        narration_text (str): The text to convert to speech
        target_duration (float): Target duration in seconds
        output_dir (str): Directory to save the audio files
        
    Returns:
        str: Path to the adjusted audio file if successful, None otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(output_dir, f'narration_audio_{timestamp}.mp3')
        adjusted_audio_path = os.path.join(output_dir, f'narration_audio_adjusted_{timestamp}.mp3')
        
        # Generate initial audio
        print("Generating new narration audio...")
        success = generate_speech(narration_text, audio_path)
        if not success:
            raise RuntimeError("Failed to generate speech audio")
        
        if success:
            # Load the generated audio to get its duration
            audio = AudioFileClip(audio_path)
            original_duration = audio.duration
            
            # Calculate the speed factor needed to match target duration
            speed_factor = original_duration / target_duration
            
            print(f"Adjusting audio speed (original duration: {original_duration:.2f}s, target: {target_duration:.2f}s)")
            
            # Create speed-adjusted audio using time transformation
            def speed_change(t):
                return speed_factor * t
                
            adjusted_audio = audio.set_make_frame(lambda t: audio.get_frame(speed_change(t)))
            adjusted_audio.duration = target_duration
            
            # Save the adjusted audio with a valid sample rate
            print(f"Saving adjusted audio to: {adjusted_audio_path}")
            adjusted_audio.write_audiofile(adjusted_audio_path, fps=44100)
            
            # Clean up
            audio.close()
            adjusted_audio.close()
            
            print(f"Successfully generated narration audio: {adjusted_audio_path}")
            return adjusted_audio_path
            
    except Exception as e:
        print(f"Error generating narration audio: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Generate narration audio for video')
    parser.add_argument('--text_file', type=str, help='Path to narration text file')
    parser.add_argument('--duration', type=float, help='Target duration in seconds')
    parser.add_argument('--output_dir', type=str, help='Output directory for audio files')
    args = parser.parse_args()
    
    # Read narration text
    with open(args.text_file, 'r') as f:
        narration_text = f.read()
    
    # Generate audio
    audio_path = generate_narration_for_video(narration_text, args.duration, args.output_dir)
    
    if audio_path:
        # Save the path to a JSON file for the main script to use
        result = {
            "narration_audio_path": audio_path,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        result_path = os.path.join(args.output_dir, "narration_result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f) 
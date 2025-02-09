import os
from datetime import datetime
from moviepy.editor import AudioFileClip
from video_generation2 import generate_narration_audio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_narration_audio():
    # Create a timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create test directory if it doesn't exist
    test_dir = f"test_generated_audio_{timestamp}"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test variables
    test_narration_text = """
    In a world where technology and humanity intersect, a remarkable story unfolds.
    Through the bustling streets and quiet moments, we witness the transformation
    of ordinary lives into extraordinary tales. Each step forward brings new
    discoveries, challenges, and unexpected connections that shape our journey.
    """
    
    target_duration = 16  # Target duration in seconds
    
    print("\nTesting narration generation and speed adjustment...")
    adjusted_audio_path = generate_narration_audio(test_narration_text, target_duration)
    
    if adjusted_audio_path and os.path.exists(adjusted_audio_path):
        print(f"✅ Test passed! Adjusted audio file generated at: {adjusted_audio_path}")
        
        # Print duration for verification
        adjusted_audio = AudioFileClip(adjusted_audio_path)
        print(f"\nTarget duration: {target_duration} seconds")
        print(f"Actual duration: {adjusted_audio.duration:.2f} seconds")
        
        # Clean up
        adjusted_audio.close()
    else:
        print("❌ Test failed! Audio file was not generated")

if __name__ == "__main__":
    test_narration_audio() 
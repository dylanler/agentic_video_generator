from elevenlabs import ElevenLabs
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def generate_speech(text, output_path="output.mp3", voice_id="JBFqnCBsd6RMkjVDRZzb"):
    """
    Generate speech from text using ElevenLabs API
    
    Args:
        text (str): The text to convert to speech
        output_path (str): Path where the audio file will be saved
        voice_id (str): ID of the voice to use
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not text.strip():  # Check if text is empty or only whitespace
        print("Error generating speech: Empty text provided")
        return False

    try:
        client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_multilingual_v2"
        )
        
        # Convert the generator to bytes
        audio_data = b"".join(audio)
        
        # Save the audio to a file
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        return True
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return False

# Example usage:
# read txt file
# with open("generated_videos/video_20250209_043032/narration_text_20250209_043032.txt", "r") as f:
#     speech_text = f.read()

# generate_speech(speech_text, "narration_audio_20250209_043032.mp3")

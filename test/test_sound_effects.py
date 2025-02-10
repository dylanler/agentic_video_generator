import os
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_sound_effect_generation():
    # Initialize ElevenLabs client
    elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))
    
    # Test prompts with different types of sound effects
    test_prompts = [
        "A gentle breeze rustling through autumn leaves, with distant bird chirps",
        "Heavy rain falling on a metal roof with occasional thunder",
        "Footsteps echoing in a large empty hallway with metallic reverb",
        "Spaceship engine humming with occasional beeping of control panels",
        "Ocean waves crashing on rocks with seagulls in the distance"
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs("test_sound_effects", exist_ok=True)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTesting sound effect {i}: {prompt}")
        output_path = f"test_sound_effects/sound_effect_{i}.mp3"
        
        try:
            # Generate sound effect
            print("Generating sound effect...")
            sound_effect_generator = elevenlabs_client.text_to_sound_effects.convert(
                text=prompt,
                duration_seconds=5.0,
                prompt_influence=0.5
            )
            
            # Save sound effect
            print(f"Saving to {output_path}")
            with open(output_path, 'wb') as f:
                for chunk in sound_effect_generator:
                    if chunk is not None:
                        f.write(chunk)
            
            print(f"✓ Successfully generated and saved sound effect {i}")
            
            # Verify file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✓ Verified file exists and has content: {os.path.getsize(output_path)} bytes")
            else:
                print("✗ File verification failed")
                
        except Exception as e:
            print(f"✗ Failed to generate sound effect {i}: {str(e)}")

if __name__ == "__main__":
    print("Starting sound effect generation test...")
    test_sound_effect_generation()
    print("\nTest complete!") 
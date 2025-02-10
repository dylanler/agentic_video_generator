import unittest
import os
from eleven_labs_tts import generate_speech

class TestElevenLabsTTS(unittest.TestCase):
    def setUp(self):
        # Create a temporary test text file
        self.test_text = "This is a test narration. Testing text-to-speech functionality."
        self.test_text_file = "test_narration.txt"
        with open(self.test_text_file, "w") as f:
            f.write(self.test_text)
        
        self.output_audio = "test_output.mp3"

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_text_file):
            os.remove(self.test_text_file)
        if os.path.exists(self.output_audio):
            os.remove(self.output_audio)

    def test_generate_speech_from_file(self):
        # Read text from file
        with open(self.test_text_file, "r") as f:
            text_content = f.read()
        
        # Generate speech
        result = generate_speech(text_content, self.output_audio)
        
        # Assert the function returned True (successful)
        self.assertTrue(result)
        # Assert the output file was created
        self.assertTrue(os.path.exists(self.output_audio))
        # Assert the output file has content (size > 0)
        self.assertGreater(os.path.getsize(self.output_audio), 0)

    def test_generate_speech_invalid_text(self):
        # Test with empty text
        result = generate_speech("", self.output_audio)
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main() 
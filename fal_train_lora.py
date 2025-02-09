import os
import json
import fal_client

class LoraTrainer:
    def __init__(self, fal_api_key):
        self.fal_api_key = fal_api_key
        fal_client.api_key = self.fal_api_key

    def on_queue_update(self, update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log["message"])

    def train_lora(self, zip_file_path, trigger_word, steps=1000):
        # Upload the zip file
        url = fal_client.upload_file(zip_file_path)

        # Call the Flux LoRA API
        result = fal_client.subscribe(
            "fal-ai/flux-lora-fast-training",
            arguments={
                "images_data_url": url,
                "create_masks": True,
                "steps": steps,
                "trigger_word": trigger_word
            },
            with_logs=True,
            on_queue_update=self.on_queue_update,
        )

        # Create a directory for the output
        output_dir = trigger_word.replace(" ", "_")
        os.makedirs(output_dir, exist_ok=True)

        # Save the output to a JSON file
        output_file = os.path.join(output_dir, f"{trigger_word}_output.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Output saved to: {output_file}")
        return result

if __name__ == "__main__":
    # Load the FAL API key from .env file
    from dotenv import load_dotenv
    load_dotenv()
    fal_api_key = os.getenv("FAL_API_KEY")

    # Example usage
    trainer = LoraTrainer(fal_api_key)
    zip_file_path = "path/to/your/images.zip"
    trigger_word = "BEANIECIRCLELOGO"
    result = trainer.train_lora(zip_file_path, trigger_word)
    print(result)
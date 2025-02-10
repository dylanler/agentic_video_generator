from google.cloud import storage
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GCPImageUploader:
    def __init__(self):
        BUCKET_NAME = os.getenv('BUCKET_NAME')
        CREDENTIALS_FILE = os.getenv('CREDENTIALS_FILE')

        # Use the specified JSON file for credentials
        self.client = storage.Client.from_service_account_json(CREDENTIALS_FILE)
        self.bucket = self.client.bucket(BUCKET_NAME)

    def upload_image(self, image_path):
        # Get the filename from the path
        filename = os.path.basename(image_path)
        
        # Create a blob object and upload the file
        blob = self.bucket.blob(filename)
        blob.upload_from_filename(image_path)
        
        # Generate a signed URL with a default expiration of 7 days
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(days=7),
            method="GET"
        )
        
        # Return the signed URL
        return url

# Example usage
if __name__ == "__main__":
    uploader = GCPImageUploader()
    image_url = uploader.upload_image("example/input.png")
    print(f"Uploaded image URL: {image_url}")

    print("Using Application Default Credentials")
    print(f"Bucket Name: {os.getenv('BUCKET_NAME')}")

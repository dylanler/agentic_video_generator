from google.cloud import storage
import os
import datetime

class GCPImageUploader:
    def __init__(self):
        BUCKET_NAME = "hackathon-bucket-demandio"
        CREDENTIALS_FILE = "demand-io-base-c29062a50662.json"

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
    image_url = uploader.upload_image("product_img/input2.png")
    print(f"Uploaded image URL: {image_url}")

    print("Using Application Default Credentials")
    print(f"Bucket Name: hackathon-bucket-demandio")

Text generation

Python Node.js Go REST

The Gemini API can generate text output when provided text, images, video, and audio as input.

This guide shows you how to generate text using the generateContent and streamGenerateContent methods. To learn about working with Gemini's vision and audio capabilities, refer to the Vision and Audio guides.

Generate text from text-only input
The simplest way to generate text using the Gemini API is to provide the model with a single text-only input, as shown in this example:


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["How does AI work?"])
print(response.text)
In this case, the prompt ("Explain how AI works") doesn't include any output examples, system instructions, or formatting information. It's a zero-shot approach. For some use cases, a one-shot or few-shot prompt might produce output that's more aligned with user expectations. In some cases, you might also want to provide system instructions to help the model understand the task or follow specific guidelines.

Generate text from text-and-image input
The Gemini API supports multimodal inputs that combine text and media files. The following example shows how to generate text from text-and-image input:


from PIL import Image
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

image = Image.open("/path/to/organ.png")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, "Tell me about this instrument"])
print(response.text)
Generate a text stream
By default, the model returns a response after completing the entire text generation process. You can achieve faster interactions by not waiting for the entire result, and instead use streaming to handle partial results.

The following example shows how to implement streaming using the streamGenerateContent method to generate text from a text-only input prompt.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"])
for chunk in response:
    print(chunk.text, end="")
Create a chat conversation
The Gemini SDK lets you collect multiple rounds of questions and responses, allowing users to step incrementally toward answers or get help with multipart problems. This SDK feature provides an interface to keep track of conversations history, but behind the scenes uses the same generateContent method to create the response.

The following code example shows a basic chat implementation:


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

chat = client.chats.create(model="gemini-2.0-flash")
response = chat.send_message("I have 2 dogs in my house.")
print(response.text)
response = chat.send_message("How many paws are in my house?")
print(response.text)
for message in chat._curated_history:
    print(f'role - ', message.role, end=": ")
    print(message.parts[0].text)
You can also use streaming with chat, as shown in the following example:


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

chat = client.chats.create(model="gemini-2.0-flash")
response = chat.send_message_stream("I have 2 dogs in my house.")
for chunk in response:
    print(chunk.text, end="")
response = chat.send_message_stream("How many paws are in my house?")
for chunk in response:
    print(chunk.text, end="")
for message in chat._curated_history:
    print(f'role - ', message.role, end=": ")
    print(message.parts[0].text)
Configure text generation
Every prompt you send to the model includes parameters that control how the model generates responses. You can use GenerationConfig to configure these parameters. If you don't configure the parameters, the model uses default options, which can vary by model.

The following example shows how to configure several of the available options.


from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"],
    config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.1
    )
)
print(response.text)
Add system instructions
System instructions let you steer the behavior of a model based on your specific needs and use cases.

By giving the model system instructions, you provide the model additional context to understand the task, generate more customized responses, and adhere to specific guidelines over the full user interaction with the model. You can also specify product-level behavior by setting system instructions, separate from prompts provided by end users.

You can set system instructions when you initialize your model:


sys_instruct="You are a cat. Your name is Neko."
client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=sys_instruct),
    contents=["your prompt here"]
)
Then, you can send requests to the model as usual.

For an interactive end to end example of using system instructions, see the system instructions colab.

What's next
Now that you have explored the basics of the Gemini API, you might want to try:

Vision understanding: Learn how to use Gemini's native vision understanding to process images and videos.
Audio understanding: Learn how to use Gemini's native audio understanding to process audio files.

Explore vision capabilities with the Gemini API

Python Node.js Go REST

Try a Colab notebook
View notebook on GitHub
Gemini models are able to process images and videos, enabling many frontier developer use cases that would have historically required domain specific models. Some of Gemini's vision capabilities include the ability to:

Caption and answer questions about images
Transcribe and reason over PDFs, including up to 2 million tokens
Describe, segment, and extract information from videos up to 90 minutes long
Detect objects in an image and return bounding box coordinates for them
Gemini was built to be multimodal from the ground up and we continue to push the frontier of what is possible.

Image input
For total image payload size less than 20MB, we recommend either uploading base64 encoded images or directly uploading locally stored image files.

Working with local images
If you are using the Python imaging library (Pillow ), you can use PIL image objects too.


from google import genai
from google.genai import types

import PIL.Image

image = PIL.Image.open('/path/to/image.png')

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["What is this image?", image])

print(response.text)
Base64 encoded images
You can upload public image URLs by encoding them as Base64 payloads. The following code example shows how to do this using only standard library tools:


from google import genai
from google.genai import types

import requests

image_path = "https://goo.gle/instrument-img"
image = requests.get(image_path)

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What is this image?",
              types.Part.from_bytes(image.content, "image/jpeg")])

print(response.text)
Multiple images
To prompt with multiple images, you can provide multiple images in the call to generate_content. These can be in any supported format, including base64 or PIL.


from google import genai
from google.genai import types

import pathlib
import PIL.Image

image_path_1 = "path/to/your/image1.jpeg"  # Replace with the actual path to your first image
image_path_2 = "path/to/your/image2.jpeg" # Replace with the actual path to your second image

image_url_1 = "https://goo.gle/instrument-img" # Replace with the actual URL to your third image

pil_image = PIL.Image.open(image_path_1)

b64_image = types.Part.from_bytes(
    pathlib.Path(image_path_2).read_bytes(), "image/jpeg")

downloaded_image = requests.get(image_url_1)

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What do these images have in common?",
              pil_image, b64_image, downloaded_image])

print(response.text)
Note that these inline data calls don't include many of the features available through the File API, such as getting file metadata, listing, or deleting files.

Large image payloads
When the combination of files and system instructions that you intend to send is larger than 20 MB in size, use the File API to upload those files.

Use the media.upload method of the File API to upload an image of any size.

Note: The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but cannot be downloaded from the API. It is available at no cost in all regions where the Gemini API is available.
After uploading the file, you can make GenerateContent requests that reference the File API URI. Select the generative model and provide it with a text prompt and the uploaded image.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

img_path = "/path/to/Cajun_instruments.jpg"
file_ref = client.files.upload(path=img_path)
print(f'{file_ref=}')

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What can you tell me about these instruments?",
              file_ref])

print(response.text)
OpenAI Compatibility
You can access Gemini's image understanding capabilities using the OpenAI libraries. This lets you integrate Gemini into existing OpenAI workflows by updating three lines of code and using your Gemini API key. See the Image understanding example for code demonstrating how to send images encoded as Base64 payloads.

Prompting with images
In this tutorial, you will upload images using the File API or as inline data and generate content based on those images.

Technical details (images)
Gemini 1.5 Pro and 1.5 Flash support a maximum of 3,600 image files.

Images must be in one of the following image data MIME types:

PNG - image/png
JPEG - image/jpeg
WEBP - image/webp
HEIC - image/heic
HEIF - image/heif
Tokens
Here's how tokens are calculated for images:

Gemini 1.0 Pro Vision: Each image accounts for 258 tokens.
Gemini 1.5 Flash and Gemini 1.5 Pro: If both dimensions of an image are less than or equal to 384 pixels, then 258 tokens are used. If one dimension of an image is greater than 384 pixels, then the image is cropped into tiles. Each tile size defaults to the smallest dimension (width or height) divided by 1.5. If necessary, each tile is adjusted so that it's not smaller than 256 pixels and not greater than 768 pixels. Each tile is then resized to 768x768 and uses 258 tokens.
Gemini 2.0 Flash: Image inputs with both dimensions <=384 pixels are counted as 258 tokens. Images larger in one or both dimensions are cropped and scaled as needed into tiles of 768x768 pixels, each counted as 258 tokens.
For best results
Rotate images to the correct orientation before uploading.
Avoid blurry images.
If using a single image, place the text prompt after the image.
Capabilities
This section outlines specific vision capabilities of the Gemini model, including object detection and bounding box coordinates.

Get a bounding box for an object
Gemini models are trained to return bounding box coordinates as relative widths or heights in the range of [0, 1]. These values are then scaled by 1000 and converted to integers. Effectively, the coordinates represent the bounding box on a 1000x1000 pixel version of the image. Therefore, you'll need to convert these coordinates back to the dimensions of your original image to accurately map the bounding boxes.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

prompt = (
  "Return a bounding box for each of the objects in this image "
  "in [ymin, xmin, ymax, xmax] format.")

response = client.models.generate_content(
  model="gemini-1.5-pro",
  contents=[sample_file_1, prompt])

print(response.text)
You can use bounding boxes for object detection and localization within images and video. By accurately identifying and delineating objects with bounding boxes, you can unlock a wide range of applications and enhance the intelligence of your projects.

Key Benefits
Simple: Integrate object detection capabilities into your applications with ease, regardless of your computer vision expertise.
Customizable: Produce bounding boxes based on custom instructions (e.g. "I want to see bounding boxes of all the green objects in this image"), without having to train a custom model.
Technical Details
Input: Your prompt and associated images or video frames.
Output: Bounding boxes in the [y_min, x_min, y_max, x_max] format. The top left corner is the origin. The x and y axis go horizontally and vertically, respectively. Coordinate values are normalized to 0-1000 for every image.
Visualization: AI Studio users will see bounding boxes plotted within the UI.
For Python developers, try the 2D spatial understanding notebook or the experimental 3D pointing notebook.

Normalize coordinates
The model returns bounding box coordinates in the format [y_min, x_min, y_max, x_max]. To convert these normalized coordinates to the pixel coordinates of your original image, follow these steps:

Divide each output coordinate by 1000.
Multiply the x-coordinates by the original image width.
Multiply the y-coordinates by the original image height.
To explore more detailed examples of generating bounding box coordinates and visualizing them on images, we encourage you to review our Object Detection cookbook example.

Prompting with video
In this tutorial, you will upload a video using the File API and generate content based on those images.

Note: The File API is required to upload video files, due to their size. However, the File API is only available for Python, Node.js, Go, and REST.
Technical details (video)
Gemini 1.5 Pro and Flash support up to approximately an hour of video data.

Video must be in one of the following video format MIME types:

video/mp4
video/mpeg
video/mov
video/avi
video/x-flv
video/mpg
video/webm
video/wmv
video/3gpp
The File API service extracts image frames from videos at 1 frame per second (FPS) and audio at 1Kbps, single channel, adding timestamps every second. These rates are subject to change in the future for improvements in inference.

Note: The details of fast action sequences may be lost at the 1 FPS frame sampling rate. Consider slowing down high-speed clips for improved inference quality.
Individual frames are 258 tokens, and audio is 32 tokens per second. With metadata, each second of video becomes ~300 tokens, which means a 1M context window can fit slightly less than an hour of video.

To ask questions about time-stamped locations, use the format MM:SS, where the first two digits represent minutes and the last two digits represent seconds.

For best results:

Use one video per prompt.
If using a single video, place the text prompt after the video.
Upload a video file using the File API
Note: The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but they cannot be downloaded using any API. It is available at no cost in all regions where the Gemini API is available.
The File API accepts video file formats directly. This example uses the short NASA film "Jupiter's Great Red Spot Shrinks and Grows". Credit: Goddard Space Flight Center (GSFC)/David Ladd (2018).

"Jupiter's Great Red Spot Shrinks and Grows" is in the public domain and does not show identifiable people. (NASA image and media usage guidelines.)

Start by retrieving the short video:


wget https://storage.googleapis.com/generativeai-downloads/images/GreatRedSpot.mp4
Upload the video using the File API and print the URI.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

print("Uploading file...")
video_file = client.files.upload(path="GreatRedSpot.mp4")
print(f"Completed upload: {video_file.uri}")
Verify file upload and check state
Verify the API has successfully received the files by calling the files.get method.

Note: Video files have a State field in the File API. When a video is uploaded, it will be in the PROCESSING state until it is ready for inference. Only ACTIVE files can be used for model inference.

import time

# Check whether the file is ready to be used.
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(1)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)

print('Done')
Prompt with a video and text
Once the uploaded video is in the ACTIVE state, you can make GenerateContent requests that specify the File API URI for that video. Select the generative model and provide it with the uploaded video and a text prompt.


from IPython.display import Markdown

# Pass the video file reference like any other media part.
response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[
        video_file,
        "Summarize this video. Then create a quiz with answer key "
        "based on the information in the video."])

# Print the response, rendering any Markdown
Markdown(response.text)
Refer to timestamps in the content
You can use timestamps of the form HH:MM:SS to refer to specific moments in the video.


prompt = "What are the examples given at 01:05 and 01:19 supposed to show us?"

response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[video_file, prompt])

print(response.text)
Transcribe video and provide visual descriptions
The Gemini models can transcribe and provide visual descriptions of video content by processing both the audio track and visual frames. For visual descriptions, the model samples the video at a rate of 1 frame per second. This sampling rate may affect the level of detail in the descriptions, particularly for videos with rapidly changing visuals.


prompt = (
    "Transcribe the audio from this video, giving timestamps for "
    "salient events in the video. Also provide visual descriptions.")

response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[video_file, prompt])

print(response.text)
List files
You can list all files uploaded using the File API and their URIs using files.list.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

print('My files:')
for f in client.files.list():
  print(" ", f'{f.name}: {f.uri}')
Delete files
Files uploaded using the File API are automatically deleted after 2 days. You can also manually delete them using files.delete.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

# Upload a file
poem_file = client.files.upload(path="poem.txt")

# Files will auto-delete after a period.
print(poem_file.expiration_time)

# Or they can be deleted explicitly.
dr = client.files.delete(name=poem_file.name)

try:
  client.models.generate_content(
      model="gemini-2.0-flash-exp",
      contents=['Finish this poem:', poem_file])
except genai.errors.ClientError as e:
  print(e.code)  # 403
  print(e.status)  # PERMISSION_DENIED
  print(e.message)  # You do not have permission to access the File .. or it may not exist.
What's next
This guide shows how to upload image and video files using the File API and then generate text outputs from image and video inputs. To learn more, see the following resources:

File prompting strategies: The Gemini API supports prompting with text, image, audio, and video data, also known as multimodal prompting.
System instructions: System instructions let you steer the behavior of the model based on your specific needs and use cases.
Safety guidance: Sometimes generative AI models produce unexpected outputs, such as outputs that are inaccurate, biased, or offensive. Post-processing and human evaluation are essential to limit the risk of harm from such outputs.

Generate structured output with the Gemini API

Python Node.js Go Dart (Flutter) Android Swift Web REST



Gemini generates unstructured text by default, but some applications require structured text. For these use cases, you can constrain Gemini to respond with JSON, a structured data format suitable for automated processing. You can also constrain the model to respond with one of the options specified in an enum.

Here are a few use cases that might require structured output from the model:

Build a database of companies by pulling company information out of newspaper articles.
Pull standardized information out of resumes.
Extract ingredients from recipes and display a link to a grocery website for each ingredient.
In your prompt, you can ask Gemini to produce JSON-formatted output, but note that the model is not guaranteed to produce JSON and nothing but JSON. For a more deterministic response, you can pass a specific JSON schema in a responseSchema field so that Gemini always responds with an expected structure.

This guide shows you how to generate JSON using the generateContent method through the SDK of your choice or using the REST API directly. The examples show text-only input, although Gemini can also produce JSON responses to multimodal requests that include images, videos, and audio.

Before you begin: Set up your project and API key
Before calling the Gemini API, you need to set up your project and configure your API key.

 Expand to view how to set up your project and API key

Generate JSON
When the model is configured to output JSON, it responds to any prompt with JSON-formatted output.

You can control the structure of the JSON response by supplying a schema. There are two ways to supply a schema to the model:

As text in the prompt
As a structured schema supplied through model configuration
Both approaches work in both Gemini 1.5 Flash and Gemini 1.5 Pro.

Supply a schema as text in the prompt
The following example prompts the model to return cookie recipes in a specific JSON format.

Since the model gets the format specification from text in the prompt, you may have some flexibility in how you represent the specification. Any reasonable format for representing a JSON schema may work.


from google import genai

prompt = """List a few popular cookie recipes in JSON format.

Use this JSON schema:

Recipe = {'recipe_name': str, 'ingredients': list[str]}
Return: list[Recipe]"""

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt,
)

# Use the response as a JSON string.
print(response.text)
The output might look like this:


```json
[
    {
        "recipe_name": "Chocolate Chip Cookies",
        "ingredients": [
            "1 cup (2 sticks) unsalted butter, softened",
            "3/4 cup granulated sugar",
            "3/4 cup packed brown sugar",
            "1 teaspoon vanilla extract",
            "2 large eggs",
            "2 1/4 cups all-purpose flour",
            "1 teaspoon baking soda",
            "1 teaspoon salt",
             "2 cups chocolate chips"
        ]
    }, ...]
```json
Supply a schema through model configuration
The following example does the following:

Instantiates a model configured through a schema to respond with JSON.
Prompts the model to return cookie recipes.
Important: This more formal method for declaring the JSON schema gives you more precise control than relying just on text in the prompt.

from google import genai
from pydantic import BaseModel, TypeAdapter


class Recipe(BaseModel):
  recipe_name: str
  ingredients: list[str]


client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='List a few popular cookie recipes.',
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Recipe],
    },
)
# Use the response as a JSON string.
print(response.text)

# Use instantiated objects.
my_recipes: list[Recipe] = response.parsed
The output might look like this:


[
 {
   "ingredients": [
     "2 1/4 cups all-purpose flour",
     "1 teaspoon baking soda",
     "1 teaspoon salt",
     "1 cup (2 sticks) unsalted butter, softened",
     "3/4 cup granulated sugar",
     "3/4 cup packed brown sugar",
     "1 teaspoon vanilla extract",
     "2 large eggs",
     "2 cups chocolate chips"
   ],
   "recipe_name": "Classic Chocolate Chip Cookies"
 }, ... ]
Schema Definition Syntax
Specify the schema for the JSON response in the response_schema property of your model configuration. The value of response_schema must be a either:

A type, as you would use in a type annotation. See the Python typing module.
An instance of genai.types.Schema.
The dict equivalent of genai.types.Schema.
Define a Schema with a Type
The easiest way to define a schema is with a direct type. This is the approach used in the preceding example:


config={'response_mime_type': 'application/json',
        'response_schema': list[Recipe]}
The Gemini API Python client library supports schemas defined with the following types (where AllowedType is any allowed type):

int
float
bool
str
list[AllowedType]
For structured types:
dict[str, AllowedType]. This annotation declares all dict values to be the same type, but doesn't specify what keys should be included.
User-defined Pydantic models. This approach lets you specify the key names and define different types for the values associated with each of the keys, including nested structures.
Use an enum to constrain output
In some cases you might want the model to choose a single option from a list of options. To implement this behavior, you can pass an enum in your schema. You can use an enum option anywhere you could use a str in the response_schema, because an enum is a list of strings. Like a JSON schema, an enum lets you constrain model output to meet the requirements of your application.

For example, assume that you're developing an application to classify musical instruments into one of five categories: "Percussion", "String", "Woodwind", "Brass", or ""Keyboard"". You could create an enum to help with this task.

In the following example, you pass the enum class Instrument as the response_schema, and the model should choose the most appropriate enum option.


from google import genai
import enum

class Instrument(enum.Enum):
  PERCUSSION = "Percussion"
  STRING = "String"
  WOODWIND = "Woodwind"
  BRASS = "Brass"
  KEYBOARD = "Keyboard"

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='What type of instrument is an oboe?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': Instrument,
    },
)

print(response.text)
# Woodwind
The Python SDK will translate the type declarations for the API. However, the API accepts a subset of the OpenAPI 3.0 schema (Schema). You can also pass the schema as JSON:


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='What type of instrument is an oboe?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': {
            "type": "STRING",
            "enum": ["Percussion", "String", "Woodwind", "Brass", "Keyboard"],
        },
    },
)

print(response.text)
# Woodwind
Beyond basic multiple choice problems, you can use an enum anywhere in a schema for JSON or function calling. For example, you could ask the model for a list of recipe titles and use a Grade enum to give each title a popularity grade:


from google import genai

import enum
from pydantic import BaseModel

class Grade(enum.Enum):
    A_PLUS = "a+"
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    F = "f"

class Recipe(BaseModel):
  recipe_name: str
  rating: Grade

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='List 10 home-baked cookies and give them grades based on tastiness.',
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Recipe],
    },
)

print(response.text)
# [{"rating": "a+", "recipe_name": "Classic Chocolate Chip Cookies"}, ...]
Was this helpful?
Jump to Content
Dream Machine API
Keys
Usage
Billing & Credits
Home
API
Status
Guides
API Reference
Changelog

Search
⌘K
Documentation
Welcome

API

Python SDK
Image Generation
Video Generation

JavaScript SDK
Rate Limits
Errors
FAQ
Powered by 
Video Generation
Suggest Edits
Installation
Python

pip install lumaai
https://pypi.org/project/lumaai/

Authentication
Get a key from https://lumalabs.ai/dream-machine/api/keys
Pass it to client sdk by either
setting LUMAAI_API_KEY
or passing auth_token to the client

Setting up client
Using LUMAAI_API_KEY env variable

Python

from lumaai import LumaAI

client = LumaAI()
Using auth_token parameter

Python

import os
from lumaai import LumaAI

client = LumaAI(
    auth_token=os.environ.get("LUMAAI_API_KEY"),
)

How do I get the video for a generation?
Right now the only supported way is via polling
The create endpoint returns an id which is an UUID V4
You can use it to poll for updates (you can see the video at generation.assets.video)
Usage Example
Python

import requests
import time
from lumaai import LumaAI

client = LumaAI()

generation = client.generations.create(
  prompt="A teddy bear in sunglasses playing electric guitar and dancing",
)
completed = False
while not completed:
  generation = client.generations.get(id=generation.id)
  if generation.state == "completed":
    completed = True
  elif generation.state == "failed":
    raise RuntimeError(f"Generation failed: {generation.failure_reason}")
  print("Dreaming")
  time.sleep(3)

video_url = generation.assets.video

# download the video
response = requests.get(video_url, stream=True)
with open(f'{generation.id}.mp4', 'wb') as file:
    file.write(response.content)
print(f"File downloaded as {generation.id}.mp4")
Async library
Import and use AsyncLumaai

Python

import os
from lumaai import AsyncLumaAI

client = AsyncLumaAI(
    auth_token=os.environ.get("LUMAAI_API_KEY"),
)
For all the functions add await (eg. below)

Python

generation = await client.generations.create(
    prompt="A teddy bear in sunglasses playing electric guitar and dancing",
)

Ray 2 Text to Video
Python

generation = client.generations.create(
    prompt="A teddy bear in sunglasses playing electric guitar and dancing",
    model="ray-2",
    resolution="720p",
    duration="5s"
)
Resolution can be 540p, 720p, 1080, 4k

Ray 2 Image to Video
Python

generation = client.generations.create(
    prompt="Low-angle shot of a majestic tiger prowling through a snowy landscape, leaving paw prints on the white blanket",
    model="ray-2",
    keyframes={
      "frame0": {
        "type": "image",
        "url": "https://storage.cdn-luma.com/dream_machine/7e4fe07f-1dfd-4921-bc97-4bcf5adea39a/video_0_thumb.jpg"
      }
    }
)

Text to Video
Python

generation = client.generations.create(
    prompt="A teddy bear in sunglasses playing electric guitar and dancing",
)
Downloading a video
Python

import requests

url = 'https://example.com/video.mp4'
response = requests.get(url, stream=True)

file_name = 'video.mp4'
with open('video.mp4', 'wb') as file:
    file.write(response.content)
print(f"File downloaded as {file_name}")
With loop, aspect ratio
Python

generation = client.generations.create(
    prompt="A teddy bear in sunglasses playing electric guitar and dancing",
    loop=True,
    aspect_ratio="3:4"
)
Image to Video
☁️
Image URL

You should upload and use your own cdn image urls, currently this is the only way to pass an image

With start frame
Python

generation = client.generations.create(
    prompt="Low-angle shot of a majestic tiger prowling through a snowy landscape, leaving paw prints on the white blanket",
    keyframes={
      "frame0": {
        "type": "image",
        "url": "https://storage.cdn-luma.com/dream_machine/7e4fe07f-1dfd-4921-bc97-4bcf5adea39a/video_0_thumb.jpg"
      }
    }
)
With start frame, loop
Python

generation = client.generations.create(
    prompt="Low-angle shot of a majestic tiger prowling through a snowy landscape, leaving paw prints on the white blanket",
    loop=True,
    keyframes={
      "frame0": {
        "type": "image",
        "url": "https://storage.cdn-luma.com/dream_machine/7e4fe07f-1dfd-4921-bc97-4bcf5adea39a/video_0_thumb.jpg"
      }
    }
)
With ending frame
Python

generation = client.generations.create(
    prompt="Low-angle shot of a majestic tiger prowling through a snowy landscape, leaving paw prints on the white blanket",
    keyframes={
      "frame1": {
        "type": "image",
        "url": "https://storage.cdn-luma.com/dream_machine/7e4fe07f-1dfd-4921-bc97-4bcf5adea39a/video_0_thumb.jpg"
      }
    }
)
With start and end keyframes
Python

generation = client.generations.create(
    prompt="Low-angle shot of a majestic tiger prowling through a snowy landscape, leaving paw prints on the white blanket",
    keyframes={
      "frame0": {
        "type": "image",
        "url": "https://storage.cdn-luma.com/dream_machine/7e4fe07f-1dfd-4921-bc97-4bcf5adea39a/video_0_thumb.jpg"
      },
      "frame1": {
        "type": "image",
        "url": "https://storage.cdn-luma.com/dream_machine/12d17326-a7b6-4538-b9b7-4a2e146d4488/video_0_thumb.jpg"
      }
    }
)
Extend Video
Extend video
Extend is currently supported only for generated videos. Please make sure the generation is in completed state before passing it

Python

generation = client.generations.create(
    prompt="A teddy bear in sunglasses playing electric guitar and dancing",
    keyframes={
      "frame0": {
        "type": "generation",
        "id": "d1968551-6113-4b46-b567-09210c2e79b0"
      }
    }
)
Reverse extend video
Generate video leading up to the provided video.

Extend is currently supported only for generated videos. Please make sure the generation is in completed state before passing it

Python

generation = client.generations.create(
    prompt="A teddy bear in sunglasses playing electric guitar and dancing",
    keyframes={
      "frame1": {
        "type": "generation",
        "id": "d1968551-6113-4b46-b567-09210c2e79b0"
      }
    }
)
Extend a video with an end-frame
Extend is currently supported only for generated videos. Please make sure the generation is in completed state before passing it

Python

generation = client.generations.create(
    prompt="Low-angle shot of a majestic tiger prowling through a snowy landscape, leaving paw prints on the white blanket",
    keyframes={
      "frame0": {
        "type": "generation",
        "id": "d1968551-6113-4b46-b567-09210c2e79b0"
      },
      "frame1": {
        "type": "image",
        "url": "https://storage.cdn-luma.com/dream_machine/12d17326-a7b6-4538-b9b7-4a2e146d4488/video_0_thumb.jpg"
      }
    }
)
Reverse extend a video with a start-frame
Extend is currently supported only for generated videos. Please make sure the generation is in completed state before passing it

Python

generation = client.generations.create(
    prompt="Low-angle shot of a majestic tiger prowling through a snowy landscape, leaving paw prints on the white blanket",
    keyframes={
      "frame0": {
        "type": "image",
        "url": "https://storage.cdn-luma.com/dream_machine/12d17326-a7b6-4538-b9b7-4a2e146d4488/video_0_thumb.jpg"
      },
      "frame1": {
        "type": "generation",
        "id": "d1968551-6113-4b46-b567-09210c2e79b0"
      }
    }
)
Interpolate between 2 videos
Interpolate is currently supported only for generated videos. Please make sure the generation is in completed state before passing it

Python

generation = client.generations.create(
    prompt="A teddy bear in sunglasses playing electric guitar and dancing",
    keyframes={
      "frame1": {
        "type": "generation",
        "id": "d312d37a-7ff4-49f2-94f8-218f3fe2a4bd"
      },
      "frame1": {
        "type": "generation",
        "id": "d1968551-6113-4b46-b567-09210c2e79b0"
      }
    }
)
Generations
Get generation with id
Python

generation = client.generations.get(id="d1968551-6113-4b46-b567-09210c2e79b0")
List all generations
Python

generation = client.generations.list(limit=100, offset=0)
Delete generation
Python

generation = client.generations.delete(id="d1968551-6113-4b46-b567-09210c2e79b0")
Camera Motions
📘
How to use camera motion

Just add the camera motion value as part of prompt itself

Get all supported camera motions
Python

supported_camera_motions = client.generations.camera_motion.list()
How to use camera motion
Camera is controlled by language in Dream Machine. You can find supported camera moves by calling the Camera Motions endpoint. This will return an array of supported camera motion strings (like "camera orbit left") which can be used in prompts. In addition to these exact strings, syntactically similar phrases also work, though there can be mismatches sometimes.


How to get a callback when generation has an update
It will get status updates (dreaming/completed/failed)
It will also get the video url as part of it when completed
It's a POST endpoint you can pass, and request body will have the generation object in it
It expected to be called multiple times for a status
If the endpoint returns a status code other than 200, it will be retried max 3 times with 100ms delay and the request has a 5s timeout
example

Python

generation = await client.generations.create(
    prompt="A teddy bear in sunglasses playing electric guitar and dancing",
    callback_url="<your_api_endpoint_here>"
)
Updated 14 days ago

Image Generation
JavaScript SDK
Did this page help you?
Table of Contents
Installation
Authentication
Setting up client
How do I get the video for a generation?
Async library
Ray 2 Text to Video
Ray 2 Image to Video
Text to Video
With loop, aspect ratio
Image to Video
With start frame
With start frame, loop
With ending frame
With start and end keyframes
Extend Video
Extend video
Reverse extend video
Extend a video with an end-frame
Reverse extend a video with a start-frame
Interpolate between 2 videos
Generations
Get generation with id
List all generations
Delete generation
Camera Motions
Get all supported camera motions
How to use camera motion
How to get a callback when generation has an update


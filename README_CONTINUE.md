# Video Generation Continuation Feature

This document explains how to use the continuation feature for video generation, which allows you to resume a video generation process that was interrupted or halted.

## Overview

The continuation feature scans a previously created video generation directory, identifies which scenes have already been generated, and continues the process from where it left off. This is particularly useful for long video generation processes that might be interrupted due to:

- Network issues
- API rate limits
- System crashes
- Power outages
- Manual interruption

## How to Use

### Command Line Option

You can use the `--continue_from_dir` option with the main `video_generation.py` script:

```bash
python video_generation.py --continue_from_dir /path/to/generated_videos/video_YYYYMMDD_HHMMSS
```

### Additional Options

When continuing a video generation, you can also specify:

- `--model`: The model to use for generating any remaining content (gemini or claude)
- `--video_engine`: The video generation engine to use (luma or ltx)
- `--skip_narration`: Skip narration generation
- `--skip_sound_effects`: Skip sound effects generation
- `--first_frame_image_gen`: Generate first frame images for each scene using Luma AI
- `--initial_image_path`: Path to local image to use as starting frame for the first video
- `--initial_image_prompt`: Prompt to generate initial image using Luma AI for the first video

Example:

```bash
python video_generation.py --continue_from_dir /path/to/generated_videos/video_YYYYMMDD_HHMMSS --model claude --video_engine luma --first_frame_image_gen
```

Note: You cannot use both `--initial_image_path` and `--initial_image_prompt` at the same time. Choose one or the other if you want to specify an initial image.

### Using the Helper Script

For convenience, you can also use the `continue_video_example.py` script, which provides a more detailed analysis of the directory before continuing:

```bash
python continue_video_example.py --directory /path/to/generated_videos/video_YYYYMMDD_HHMMSS
```

This script supports all the same options as the main script:

```bash
python continue_video_example.py --directory /path/to/generated_videos/video_YYYYMMDD_HHMMSS --model claude --first_frame_image_gen
```

The script will:
1. Scan the directory and show what has been generated so far
2. Display which scenes are complete and which are incomplete
3. Automatically continue the generation process if needed

#### Listing JSON Files

If you want to see detailed information about all JSON files in the directory, you can use the `--list_json_files` option:

```bash
python continue_video_example.py --directory /path/to/generated_videos/video_YYYYMMDD_HHMMSS --list_json_files
```

This will show:
- The name of each JSON file
- The type of data it contains (scenes with environments, scene metadata, physical environments, etc.)
- The file size
- The number of items in the file

## Image Generation Options

When continuing video generation, you have several options for handling the first frame of videos:

### First Frame Image Generation

The `--first_frame_image_gen` option will generate a new first frame image for each scene using Luma AI. This can help maintain visual consistency when generating new scenes to complete an interrupted video.

```bash
python video_generation.py --continue_from_dir /path/to/dir --first_frame_image_gen
```

### Initial Image Options

You can provide an initial image for the first video in one of two ways:

1. **Local Image Path**: Use an existing image file as the starting frame
   ```bash
   python video_generation.py --continue_from_dir /path/to/dir --initial_image_path /path/to/image.jpg
   ```

2. **Image Generation Prompt**: Generate an image using Luma AI based on a text prompt
   ```bash
   python video_generation.py --continue_from_dir /path/to/dir --initial_image_prompt "A serene forest at dawn with mist rising from the ground"
   ```

These options can be particularly useful when you need to maintain visual consistency between the previously generated scenes and the new ones.

## How It Works

The continuation process works as follows:

1. **Directory Scanning**: The system scans the specified directory to find:
   - The scenes JSON file (specifically looking for `scenes_TIMESTAMP.json`)
   - Completed scene videos
   - Sound effect files
   - Narration text and audio
   - Any final video

2. **Scene Analysis**: The system determines which scenes have been completed and which still need to be generated.

3. **Continuation**:
   - If all scenes are already generated, the system will just stitch the videos together
   - If some scenes are missing, the system will generate only those scenes
   - The system will then combine all videos (previously generated and newly generated) in the correct order

4. **Final Output**: The system produces a final video with all scenes, sound effects, and narration.

## JSON File Selection

The system needs to find the correct scenes JSON file to continue the generation process. It follows these steps:

1. First, it looks for a file named `scenes_TIMESTAMP.json` where TIMESTAMP matches the timestamp in the directory name
2. If that's not found, it looks for any file that matches the pattern `scenes_YYYYMMDD_HHMMSS.json`
3. If multiple matching files are found, it selects the most recently modified one
4. As a last resort, it examines all JSON files in the directory to find one that contains scene data

This ensures that the system can find the correct scene data even if the directory structure is not exactly as expected.

## Troubleshooting

If you encounter issues with the continuation feature:

1. **Missing Scenes JSON**: Ensure the directory contains a valid scenes JSON file (usually named `scenes_YYYYMMDD_HHMMSS.json`)
   - Use the `--list_json_files` option to see all JSON files in the directory
   - Check if any of them contain scene data

2. **Inconsistent Scene Numbers**: Make sure scene numbers in the JSON file match the scene numbers in the video filenames

3. **Corrupted Videos**: If any generated videos are corrupted, you may need to delete them so they can be regenerated

4. **Timestamp Issues**: If the timestamp in the directory name doesn't match the timestamps in the files, the system might have trouble identifying the correct files

5. **Image Generation Issues**: If using `--first_frame_image_gen` or `--initial_image_prompt`, ensure that the Luma AI API key is properly set in your environment variables

## Example Directory Structure

A typical video generation directory might look like:

```
generated_videos/video_20240601_123456/
├── scenes_20240601_123456.json                 # Main scenes JSON file with environments
├── scene_metadata_no_env_20240601_123456.json  # Scene metadata without environments
├── scene_physical_environment_20240601_123456.json  # Physical environments only
├── narration_text_20240601_123456.txt
├── narration_audio_20240601_123456.mp3
├── narration_audio_adjusted_20240601_123456.mp3
├── scene_1_20240601_123456.mp4
├── scene_2_20240601_123456.mp4
├── scene_3_20240601_123456.mp4
└── scene_1_all_vid_20240601_123456/
    ├── scene_1_vid_1_20240601_123456.mp4
    ├── scene_1_vid_2_20240601_123456.mp4
    └── scene_1_sound.mp3
```

The continuation feature will analyze this structure to determine what has been completed and what still needs to be done. 
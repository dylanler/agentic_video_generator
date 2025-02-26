#!/usr/bin/env python3
"""
Example script demonstrating how to continue video generation from a previously interrupted process.
"""

import os
import argparse
import json
from scan_directory import scan_directory

def main():
    parser = argparse.ArgumentParser(description='Continue video generation from a previously interrupted process')
    parser.add_argument('--directory', type=str, required=True,
                       help='Directory containing the interrupted video generation process')
    parser.add_argument('--model', type=str, choices=['gemini', 'claude'], default='gemini',
                       help='Model to use for scene generation (default: gemini)')
    parser.add_argument('--video_engine', type=str, choices=['luma', 'ltx'], default='luma',
                       help='Video generation engine to use (default: luma)')
    parser.add_argument('--skip_narration', action='store_true',
                       help='Skip narration generation')
    parser.add_argument('--skip_sound_effects', action='store_true',
                       help='Skip sound effects generation')
    parser.add_argument('--list_json_files', action='store_true',
                       help='List all JSON files in the directory with their content summary')
    parser.add_argument('--first_frame_image_gen', action='store_true',
                       help='Generate first frame images for each scene using Luma AI')
    parser.add_argument('--initial_image_path', type=str,
                       help='Path to local image to use as starting frame for the first video')
    parser.add_argument('--initial_image_prompt', type=str,
                       help='Prompt to generate initial image using Luma AI for the first video')
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return

    if args.initial_image_path and args.initial_image_prompt:
        print("Error: Cannot provide both initial_image_path and initial_image_prompt. Please choose one.")
        return

    # List all JSON files in the directory if requested
    if args.list_json_files:
        print("\n=== JSON Files in Directory ===")
        json_files = [f for f in os.listdir(args.directory) if f.endswith('.json')]
        
        if not json_files:
            print("No JSON files found in the directory.")
        else:
            for json_file in json_files:
                file_path = os.path.join(args.directory, json_file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Determine the type of JSON file
                    file_type = "Unknown"
                    if isinstance(data, list):
                        if len(data) > 0:
                            if 'scene_number' in data[0] and 'scene_physical_environment' in data[0]:
                                file_type = "Scenes with environments"
                            elif 'scene_number' in data[0]:
                                file_type = "Scene metadata"
                            elif 'scene_physical_environment' in data[0]:
                                file_type = "Physical environments"
                    
                    # Get file size
                    file_size = os.path.getsize(file_path) / 1024  # Size in KB
                    
                    print(f"File: {json_file}")
                    print(f"  Type: {file_type}")
                    print(f"  Size: {file_size:.2f} KB")
                    print(f"  Items: {len(data) if isinstance(data, list) else 'Not a list'}")
                    print()
                    
                except Exception as e:
                    print(f"File: {json_file}")
                    print(f"  Error reading file: {str(e)}")
                    print()
        
        print("=== End of JSON Files List ===\n")

    # Scan the directory to see what's already been generated
    try:
        scan_result = scan_directory(args.directory)
        print(f"Directory: {scan_result['directory']}")
        print(f"Timestamp: {scan_result['timestamp']}")
        
        # Print information about the scenes JSON file
        if scan_result['scenes_json_path']:
            print(f"Scenes JSON: {os.path.basename(scan_result['scenes_json_path'])}")
        else:
            print("Scenes JSON: Not found")
        
        print(f"Total scenes: {len(scan_result['scenes_data']) if scan_result['scenes_data'] else 0}")
        print(f"Completed scenes: {scan_result['completed_scenes']}")
        print(f"Incomplete scenes: {scan_result['incomplete_scenes']}")
        print(f"Narration text: {'Found' if scan_result['narration_text_path'] else 'Not found'}")
        print(f"Narration audio: {'Found' if scan_result['narration_audio_path'] else 'Not found'}")
        print(f"Final video: {'Found' if scan_result['final_video_path'] else 'Not found'}")
        
        # If there are incomplete scenes, continue the generation
        if scan_result['incomplete_scenes']:
            print("\nContinuing video generation...")
            
            # Build the command to continue video generation
            cmd = [
                "python", "video_generation.py",
                "--continue_from_dir", args.directory,
                "--model", args.model,
                "--video_engine", args.video_engine
            ]
            
            if args.skip_narration:
                cmd.append("--skip_narration")
            
            if args.skip_sound_effects:
                cmd.append("--skip_sound_effects")
            
            if args.first_frame_image_gen:
                cmd.append("--first_frame_image_gen")
            
            if args.initial_image_path:
                cmd.extend(["--initial_image_path", args.initial_image_path])
            
            if args.initial_image_prompt:
                cmd.extend(["--initial_image_prompt", args.initial_image_prompt])
            
            # Print the command
            print("Running command:", " ".join(cmd))
            
            # Execute the command
            import subprocess
            subprocess.run(cmd)
        else:
            print("\nAll scenes are already generated. You can run the following command to stitch the videos:")
            cmd = [
                "python", "video_generation.py",
                "--continue_from_dir", args.directory,
                "--model", args.model,
                "--video_engine", args.video_engine
            ]
            
            if args.skip_narration:
                cmd.append("--skip_narration")
            
            if args.skip_sound_effects:
                cmd.append("--skip_sound_effects")
            
            print("Command:", " ".join(cmd))
    
    except Exception as e:
        print(f"Error scanning directory: {str(e)}")

if __name__ == "__main__":
    main() 
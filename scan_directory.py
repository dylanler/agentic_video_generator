import os
import json
import re
from typing import Dict, List, Tuple, Optional

def scan_directory(directory_path: str) -> Dict:
    """
    Scan a video generation directory to determine the state of the generation process.
    
    Args:
        directory_path: Path to the directory containing video generation files
        
    Returns:
        Dictionary containing information about the state of the generation process
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist")
    
    result = {
        "directory": directory_path,
        "scenes_json_path": None,
        "scenes_data": None,
        "completed_scenes": [],
        "incomplete_scenes": [],
        "narration_text_path": None,
        "narration_audio_path": None,
        "final_video_path": None,
        "timestamp": None
    }
    
    # Extract timestamp from directory name if possible
    dir_name = os.path.basename(directory_path)
    timestamp_match = re.search(r'video_(\d{8}_\d{6})', dir_name)
    if timestamp_match:
        result["timestamp"] = timestamp_match.group(1)
    
    # Find scenes JSON file - specifically look for "scenes_{timestamp}.json"
    # First try with the timestamp if available
    if result["timestamp"]:
        expected_json_name = f"scenes_{result['timestamp']}.json"
        expected_json_path = os.path.join(directory_path, expected_json_name)
        if os.path.exists(expected_json_path):
            result["scenes_json_path"] = expected_json_path
    
    # If not found with timestamp, look for any file that matches the pattern exactly
    if not result["scenes_json_path"]:
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        scenes_json_files = [f for f in json_files if re.match(r'^scenes_\d{8}_\d{6}\.json$', f)]
        
        if scenes_json_files:
            # Sort by modification time (newest first) to get the most recent one
            scenes_json_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory_path, x)), reverse=True)
            result["scenes_json_path"] = os.path.join(directory_path, scenes_json_files[0])
            
            # Extract timestamp from the filename if we didn't get it from the directory
            if not result["timestamp"]:
                timestamp_match = re.search(r'scenes_(\d{8}_\d{6})\.json', scenes_json_files[0])
                if timestamp_match:
                    result["timestamp"] = timestamp_match.group(1)
    
    # Load the scenes data if we found the JSON file
    if result["scenes_json_path"]:
        try:
            with open(result["scenes_json_path"], 'r') as f:
                result["scenes_data"] = json.load(f)
            print(f"Successfully loaded scenes data from: {result['scenes_json_path']}")
        except Exception as e:
            print(f"Error loading scenes JSON: {str(e)}")
    else:
        print("Warning: Could not find scenes JSON file. Looking for any JSON file that might contain scene data.")
        # Last resort: try any JSON file that might contain scene data
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(directory_path, json_file), 'r') as f:
                    data = json.load(f)
                    # Check if this looks like scene data (has scene_number, scene_name, etc.)
                    if isinstance(data, list) and len(data) > 0 and 'scene_number' in data[0]:
                        result["scenes_data"] = data
                        result["scenes_json_path"] = os.path.join(directory_path, json_file)
                        print(f"Found scene data in: {json_file}")
                        break
            except Exception:
                continue
    
    # Find narration files
    narration_text_files = [f for f in os.listdir(directory_path) if f.endswith('.txt') and 'narration_text_' in f]
    if narration_text_files:
        result["narration_text_path"] = os.path.join(directory_path, narration_text_files[0])
    
    narration_audio_files = [f for f in os.listdir(directory_path) if f.endswith('.mp3') and 'narration_audio_' in f]
    if narration_audio_files:
        # Prefer adjusted audio if available
        adjusted_files = [f for f in narration_audio_files if 'adjusted' in f]
        if adjusted_files:
            result["narration_audio_path"] = os.path.join(directory_path, adjusted_files[0])
        else:
            result["narration_audio_path"] = os.path.join(directory_path, narration_audio_files[0])
    
    # Find final video if it exists
    final_video_files = [f for f in os.listdir(directory_path) if f.endswith('.mp4') and 'final_video_' in f]
    if final_video_files:
        result["final_video_path"] = os.path.join(directory_path, final_video_files[0])
    
    # Check which scenes have been completed
    if result["scenes_data"]:
        for scene in result["scenes_data"]:
            scene_number = scene.get("scene_number")
            if scene_number is None:
                continue
            
            # Check if scene video exists
            scene_video_pattern = f"scene_{scene_number}_"
            scene_videos = [f for f in os.listdir(directory_path) if f.endswith('.mp4') and scene_video_pattern in f]
            
            if scene_videos:
                result["completed_scenes"].append(scene_number)
            else:
                result["incomplete_scenes"].append(scene_number)
    
    return result

def get_remaining_scenes(scan_result: Dict) -> List[Dict]:
    """
    Get the list of scenes that still need to be generated.
    
    Args:
        scan_result: Result from scan_directory function
        
    Returns:
        List of scene data for scenes that need to be generated
    """
    if not scan_result["scenes_data"]:
        return []
    
    remaining_scenes = []
    for scene in scan_result["scenes_data"]:
        scene_number = scene.get("scene_number")
        if scene_number in scan_result["incomplete_scenes"]:
            remaining_scenes.append(scene)
    
    return remaining_scenes

def get_completed_scene_videos(scan_result: Dict) -> List[str]:
    """
    Get the list of completed scene video file paths.
    
    Args:
        scan_result: Result from scan_directory function
        
    Returns:
        List of file paths to completed scene videos
    """
    if not scan_result["completed_scenes"]:
        return []
    
    directory = scan_result["directory"]
    completed_videos = []
    
    for scene_number in scan_result["completed_scenes"]:
        scene_video_pattern = f"scene_{scene_number}_"
        scene_videos = [f for f in os.listdir(directory) if f.endswith('.mp4') and scene_video_pattern in f]
        
        if scene_videos:
            # Sort by timestamp if available
            scene_videos.sort(reverse=True)  # Most recent first
            completed_videos.append(os.path.join(directory, scene_videos[0]))
    
    # Sort videos by scene number
    completed_videos.sort(key=lambda x: int(re.search(r'scene_(\d+)_', os.path.basename(x)).group(1)))
    
    return completed_videos

def get_sound_effect_files(scan_result: Dict) -> List[Optional[str]]:
    """
    Get the list of sound effect file paths for completed scenes.
    
    Args:
        scan_result: Result from scan_directory function
        
    Returns:
        List of file paths to sound effect files (None for scenes without sound effects)
    """
    if not scan_result["completed_scenes"]:
        return []
    
    directory = scan_result["directory"]
    sound_files = []
    
    for scene_number in scan_result["completed_scenes"]:
        # Look for sound effect files in scene subdirectories
        scene_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and f"scene_{scene_number}_" in d]
        
        sound_file = None
        for scene_dir in scene_dirs:
            scene_dir_path = os.path.join(directory, scene_dir)
            sound_files_in_dir = [f for f in os.listdir(scene_dir_path) if f.endswith('.mp3') and f"scene_{scene_number}_sound" in f]
            
            if sound_files_in_dir:
                sound_file = os.path.join(scene_dir_path, sound_files_in_dir[0])
                break
        
        sound_files.append(sound_file)
    
    return sound_files 
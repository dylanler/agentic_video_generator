import os
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import argparse
from datetime import datetime

def find_matching_files(directory):
    """Find all mp4 files and their corresponding mp3 files in the directory."""
    video_files = []
    sound_files = []
    
    # Get all mp4 files with their creation times
    files_with_time = []
    for file in os.listdir(directory):
        if file.endswith('.mp4'):
            file_path = os.path.join(directory, file)
            # Get file creation time
            creation_time = os.path.getctime(file_path)
            files_with_time.append((file, creation_time))
    
    # Sort files by creation time
    sorted_files = sorted(files_with_time, key=lambda x: x[1])
    
    # Process sorted files
    for file, _ in sorted_files:
        base_name = file.rsplit('.', 1)[0]
        video_path = os.path.join(directory, file)
        sound_path = os.path.join(directory, f"{base_name}_sound.mp3")
        
        video_files.append(video_path)
        # Only add sound file if it exists
        sound_files.append(sound_path if os.path.exists(sound_path) else None)
    
    return video_files, sound_files

def stitch_videos(video_files, sound_effect_files, output_dir=None, ignore_audio=False):
    """Stitch together videos with their corresponding sound effects."""
    if not output_dir:
        output_dir = os.path.dirname(video_files[0])
    
    final_clips = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Processing {len(video_files)} video files...")
    
    for i, (video_file, sound_file) in enumerate(zip(video_files, sound_effect_files), 1):
        try:
            print(f"\nProcessing video {i}/{len(video_files)}: {os.path.basename(video_file)}")
            video_clip = VideoFileClip(video_file)
            
            if not ignore_audio and sound_file and os.path.exists(sound_file):
                try:
                    print(f"Adding sound from: {os.path.basename(sound_file)}")
                    # Load sound effect
                    audio_clip = AudioFileClip(sound_file)
                    
                    print(f"Video duration: {video_clip.duration:.2f}s")
                    print(f"Audio duration: {audio_clip.duration:.2f}s")
                    
                    # If audio is longer than video, trim it
                    if audio_clip.duration > video_clip.duration:
                        print("Trimming audio to match video duration")
                        audio_clip = audio_clip.subclip(0, video_clip.duration)
                    # If audio is shorter than video, trim video
                    elif audio_clip.duration < video_clip.duration:
                        print("Trimming video to match audio duration")
                        video_clip = video_clip.subclip(0, audio_clip.duration)
                    
                    # Combine video with sound effect
                    video_clip = video_clip.set_audio(audio_clip)
                    print("Successfully combined video and audio")
                except Exception as e:
                    print(f"Warning: Failed to process audio for {sound_file}: {str(e)}")
            else:
                print("No sound file found, using video without audio")
            
            final_clips.append(video_clip)
            print(f"Successfully added clip {i} to final sequence")
        except Exception as e:
            print(f"Warning: Failed to process video {video_file}: {str(e)}")
            continue
    
    if not final_clips:
        raise RuntimeError("No video clips were successfully processed")
    
    print("\nConcatenating all clips...")
    final_clip = concatenate_videoclips(final_clips)
    
    output_path = os.path.join(output_dir, f"stitched_video_{timestamp}.mp4")
    print(f"\nWriting final video to: {output_path}")
    final_clip.write_videofile(output_path)
    
    # Close all clips
    print("\nCleaning up...")
    for clip in final_clips:
        clip.close()
    
    print("Done!")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Stitch together videos with their corresponding sound effects')
    parser.add_argument('input_dir', help='Directory containing the mp4 and mp3 files')
    parser.add_argument('--output-dir', help='Directory to save the final video (default: same as input)')
    parser.add_argument('--ignore-audio', action='store_true', help='Ignore audio files and use videos without sound')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist")
        return
    
    video_files, sound_files = find_matching_files(args.input_dir)
    
    if not video_files:
        print(f"No mp4 files found in {args.input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    for i, (v, s) in enumerate(zip(video_files, sound_files), 1):
        print(f"{i}. Video: {os.path.basename(v)}")
        print(f"   Audio: {os.path.basename(s) if s else 'None'}")
    
    try:
        output_path = stitch_videos(video_files, sound_files, args.output_dir, args.ignore_audio)
        print(f"\nSuccessfully created stitched video: {output_path}")
    except Exception as e:
        print(f"\nError: Failed to stitch videos: {str(e)}")

if __name__ == "__main__":
    main() 
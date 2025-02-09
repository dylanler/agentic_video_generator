import os
from moviepy.editor import VideoFileClip
import cv2
import sys

def test_moviepy_extraction(video_path):
    """Test frame extraction using MoviePy"""
    print("\nTesting MoviePy extraction...")
    try:
        video = VideoFileClip(video_path)
        print(f"Video duration: {video.duration} seconds")
        print(f"Video size: {video.size}")
        print(f"FPS: {video.fps}")
        
        output_path = "last_frame_moviepy.jpg"
        video.save_frame(output_path, t=video.duration-(1/24))  # Extract last frame at 24fps
        print(f"Successfully saved frame to: {output_path}")
        
        video.close()
        return True
    except Exception as e:
        print(f"MoviePy extraction failed: {str(e)}")
        return False

def test_opencv_extraction(video_path):
    """Test frame extraction using OpenCV"""
    print("\nTesting OpenCV extraction...")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Total frames: {frame_count}")
        print(f"FPS: {fps}")
        
        # Seek to last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count-1)
        ret, frame = cap.read()
        
        if ret:
            cv2.imwrite("last_frame_opencv.jpg", frame)
            print("Successfully saved frame to: last_frame_opencv.jpg")
            cap.release()
            return True
        else:
            print("Failed to read last frame")
            cap.release()
            return False
    except Exception as e:
        print(f"OpenCV extraction failed: {str(e)}")
        return False

def print_video_info(video_path):
    """Print basic video file information"""
    print("\nVideo File Information:")
    print(f"File path: {video_path}")
    print(f"File exists: {os.path.exists(video_path)}")
    print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_last_frame.py <video_file_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Print video information
    print_video_info(video_path)
    
    # Test both methods
    moviepy_success = test_moviepy_extraction(video_path)
    opencv_success = test_opencv_extraction(video_path)
    
    # Summary
    print("\nResults Summary:")
    print(f"MoviePy extraction: {'Success' if moviepy_success else 'Failed'}")
    print(f"OpenCV extraction: {'Success' if opencv_success else 'Failed'}")

if __name__ == "__main__":
    main()
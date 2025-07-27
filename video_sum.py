[6:05 pm, 27/7/2025] Sundari: Resend me pls 😭
[6:06 pm, 27/7/2025] Likitha AI&DS: import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

def extract_keyframes(video_path, threshold=30, display=False):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    keyframe_timestamps = []
    last_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
        
        if display and frame_count % 500 == 0:
            print(f"Processing frame {frame_count}")
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        if last_frame is None:
            keyframes.append(frame)
            keyframe_timestamps.append(timestamp)
            last_frame = gray_frame
            continue
            
        frame_diff = cv2.absdiff(last_frame, gray_frame)
        non_zero_count = np.count_nonzero(frame_diff)
        
        if non_zero_count > threshold:
            keyframes.append(frame)
            keyframe_timestamps.append(timestamp)
            last_frame = gray_frame
            if display:
                print(f"Frame {frame_count}: Keyframe selected")
    
    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {frame_count} total frames")
    return keyframes, keyframe_timestamps

def extract_audio_segments(video_path, keyframe_timestamps, margin=1.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    segments = []
    for timestamp in keyframe_timestamps:
        start_time = max(0, timestamp - margin)
        end_time = min(duration, timestamp + margin)
        segments.append((start_time, end_time))
    
    # Merge overlapping
    segments.sort()
    merged_segments = []
    if segments:
        current_start, current_end = segments[0]
        for start, end in segments[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        merged_segments.append((current_start, current_end))
    
    return merged_segments

def score_keyframes(keyframes):
    scores = []
    for frame in keyframes:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        score = 0.6 * brightness + 0.4 * contrast
        scores.append(score)
    return scores

def create_summary_video(input_video, output_video, segments, target_duration=None):
    print(f"Creating summary with {len(segments)} segments...")
    video = VideoFileClip(input_video)
    
    if target_duration is None:
        target_duration = video.duration * 0.25  # default 25%
    
    subclips = []
    total_duration = 0
    
    for start, end in segments:
        try:
            subclip = video.subclip(start, end)
            subclips.append(subclip)
            total_duration += (end - start)
        except Exception as e:
            print(f"Skipping segment {start}-{end}: {e}")
    
    if total_duration > target_duration and subclips:
        keep_ratio = target_duration / total_duration
        num_clips_to_keep = max(1, int(len(subclips) * keep_ratio))
        indices = np.linspace(0, len(subclips)-1, num_clips_to_keep, dtype=int)
        subclips = [subclips[i] for i in indices]
        total_duration = sum(clip.duration for clip in subclips)
    
    if subclips:
        final_clip = concatenate_videoclips(subclips)
        print(f"Writing summary video ({total_duration:.2f}s)...")
        final_clip.write_videofile(output_video, codec='libx264', audio_codec='aac')
        
        final_clip.close()
        for clip in subclips:
            clip.close()
        video.close()
        
        print(f"Summary saved to {output_video}")
        return total_duration
    else:
        print("No valid clips to include.")
        video.close()
        return 0

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": original_duration
    }

def summarize_video_simplified():
    print("=== Video Summarization ===")

    # Input path
    input_video = input("Enter input video path (e.g., /content/video.mp4): ").strip()
    if not os.path.exists(input_video):
        print(f"Error: '{input_video}' not found.")
        return

    # Output path
    output_video = input("Enter output video path (e.g., /content/summary.mp4): ").strip()
    if not output_video.endswith(".mp4"):
        print("Output should end with '.mp4'")
        return

    # Get video info
    video_info = get_video_info(input_video)
    original_duration = video_info["duration"]
    print(f"Video duration: {original_duration:.2f} seconds ({original_duration/60:.2f} minutes)")

    # Get desired summary length in minutes
    while True:
        try:
            summary_minutes = float(input("How many minutes should the summary be? ").strip())
            summary_duration = summary_minutes * 60
            if summary_duration >= original_duration:
                print("Summary duration must be less than the original video duration.")
            else:
                break
        except ValueError:
            print("Please enter a valid number (e.g., 2.5).")

    # Optional params
    threshold_input = input("Frame difference threshold (default 30): ").strip()
    threshold = int(threshold_input) if threshold_input else 30

    display_input = input("Show frame processing progress? (y/n, default y): ").strip().lower()
    display = (display_input != "n")

    # Process
    print("\n[1] Extracting keyframes...")
    keyframes, timestamps = extract_keyframes(input_video, threshold, display)

    if not keyframes:
        print("No keyframes found.")
        return

    print("\n[2] Scoring keyframes...")
    scores = score_keyframes(keyframes)

    print("\n[3] Selecting top keyframes...")
    sorted_indices = np.argsort(scores)[::-1]
    top_n = max(5, int(len(keyframes) * 0.5))
    selected_timestamps = [timestamps[i] for i in sorted_indices[:top_n]]

    print("\n[4] Creating audio-visual segments...")
    segments = extract_audio_segments(input_video, selected_timestamps, margin=1.5)
    print(f"{len(segments)} segments created.")

    print("\n[5] Creating summary video...")
    actual_duration = create_summary_video(input_video, output_video, segments, target_duration=summary_duration)

    print("\n=== Summary Report ===")
    print(f"Original duration : {original_duration:.2f} sec")
    print(f"Target summary    : {summary_duration:.2f} sec")
    print(f"Final summary     : {actual_duration:.2f} sec")
    print(f"Compression       : {100 - (actual_duration / original_duration * 100):.1f}%")
    print(f"Output saved at   : {output_video}")

# Run the simplified version
if _name_ == "_main_":
    summarize_video_simplified(
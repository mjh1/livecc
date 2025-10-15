"""
LiveCC CLI for streaming live video (webcam, RTSP, etc.)
Usage:
    python demo/cli_streaming.py --source 0  # for webcam
    python demo/cli_streaming.py --source rtsp://your-stream-url  # for RTSP stream
"""
import json
import time
import argparse
import threading
import queue
import numpy as np
import cv2
import torch
import sys
import os
from collections import deque
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.infer_streaming import LiveCCStreamingInfer
from livecc_utils.video_process_patch import smart_resize, IMAGE_FACTOR, VIDEO_MIN_PIXELS
from torchvision import transforms


class LiveFrameBuffer:
    """Buffer to store live video frames with timestamps"""
    
    def __init__(self, max_pixels=384 * 28 * 28, fps=2, buffer_size=100):
        self.frames = deque(maxlen=buffer_size)  # Store (timestamp, frame) tuples
        self.fps = fps
        self.max_pixels = max_pixels
        self.resized_height = None
        self.resized_width = None
        self.running = False
        self.frame_lock = threading.Lock()
        
    def add_frame(self, frame, timestamp):
        """Add a frame to the buffer"""
        with self.frame_lock:
            # Resize frame on first addition
            if self.resized_height is None:
                height, width = frame.shape[:2]
                self.resized_height, self.resized_width = smart_resize(
                    height, width,
                    factor=IMAGE_FACTOR,
                    min_pixels=VIDEO_MIN_PIXELS,
                    max_pixels=self.max_pixels,
                )
            
            # Resize frame
            resized_frame = cv2.resize(frame, (self.resized_width, self.resized_height))
            # Convert BGR to RGB
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            self.frames.append((timestamp, resized_frame))
    
    def get_frames_in_range(self, start_time, end_time):
        """Get all frames between start_time and end_time"""
        with self.frame_lock:
            frames_in_range = []
            timestamps = []
            
            for timestamp, frame in self.frames:
                if start_time <= timestamp <= end_time:
                    frames_in_range.append(frame)
                    timestamps.append(timestamp)
            
            if not frames_in_range:
                return None, None
            
            # Convert to torch tensor [T, H, W, C] -> [T, C, H, W]
            clip = torch.from_numpy(np.stack(frames_in_range)).permute(0, 3, 1, 2)
            
            return clip, torch.tensor(timestamps)
    
    def get_current_time(self):
        """Get the timestamp of the most recent frame"""
        with self.frame_lock:
            if self.frames:
                return self.frames[-1][0]
            return 0.0
    
    def get_all_timestamps(self):
        """Get all available timestamps"""
        with self.frame_lock:
            return torch.tensor([t for t, _ in self.frames])


class LiveStreamCapture:
    """Capture live video from webcam or stream"""
    
    def __init__(self, source, target_fps=2):
        """
        Args:
            source: Video source (0 for webcam, or RTSP URL)
            target_fps: Target frames per second to capture
        """
        self.source = source
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.buffer = None
        self.running = False
        self.capture_thread = None
        
    def start(self, frame_buffer):
        """Start capturing frames"""
        self.buffer = frame_buffer
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
    def stop(self):
        """Stop capturing frames"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
    
    def _capture_loop(self):
        """Main capture loop (runs in separate thread)"""
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")
        
        print(f"Started capturing from {self.source}")
        start_time = time.time()
        last_capture_time = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            current_time = time.time() - start_time
            
            # Only capture at target FPS
            if current_time - last_capture_time >= self.frame_interval:
                self.buffer.add_frame(frame, current_time)
                last_capture_time = current_time
                print(f"Captured frame at {current_time:.2f}s")
            
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
        
        cap.release()
        print("Stopped capturing")


def save_screenshot(frame_buffer, timestamp, output_dir="screenshots"):
    """Save a screenshot from the buffer at the given timestamp"""
    os.makedirs(output_dir, exist_ok=True)
    
    with frame_buffer.frame_lock:
        frames_list = list(frame_buffer.frames)
        # Find the closest frame to the timestamp
        closest_frame = None
        min_diff = float('inf')
        
        for t, frame in frames_list:
            diff = abs(t - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame
        
        if closest_frame is not None:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(closest_frame, cv2.COLOR_RGB2BGR)
            filename = f"{output_dir}/frame_{timestamp:.2f}s.jpg"
            cv2.imwrite(filename, frame_bgr)
            return filename
    return None


def live_cc_streaming(
    infer,
    frame_buffer,
    query="Please describe the video.",
    max_pixels=384 * 28 * 28,
    save_screenshots=False,
    screenshot_dir="screenshots",
    **kwargs
):
    """
    Process live video stream using LiveCC
    
    Args:
        infer: LiveCCDemoInfer instance
        frame_buffer: LiveFrameBuffer instance
        query: Query to ask about the video
        max_pixels: Maximum pixels for video processing
        save_screenshots: Whether to save screenshot at start of each segment
        screenshot_dir: Directory to save screenshots
    """
    state = {
        'last_timestamp': -1 / infer.fps,
        'last_video_pts_index': -1,
    }
    
    commentaries = []
    
    while True:
        current_time = frame_buffer.get_current_time()
        
        # Wait until we have enough buffer
        if current_time < infer.initial_time_interval:
            time.sleep(0.5)
            continue
        
        # Update state with current video timestamp
        state['video_timestamp'] = current_time
        state['video_pts'] = frame_buffer.get_all_timestamps()
        state['frame_buffer'] = frame_buffer
        
        # Process frames
        for (start_t, stop_t), response, state in infer.live_cc_streaming(
            message=query,
            state=state,
            max_pixels=max_pixels,
            streaming_eos_base_threshold=None,  # Disable streaming EOS to get full responses
            streaming_eos_threshold_step=None,
            **kwargs
        ):
            # Save screenshot at the start of each segment
            if save_screenshots:
                screenshot_path = save_screenshot(frame_buffer, start_t, screenshot_dir)
                if screenshot_path:
                    print(f'[SCREENSHOT] Saved: {screenshot_path}')
            
            print(f'{start_t:.1f}s-{stop_t:.1f}s: {response}')
            commentaries.append([start_t, stop_t, response])
        
        # Sleep a bit before next iteration
        time.sleep(0.5)
    
    return commentaries


def parse_args():
    parser = argparse.ArgumentParser(description='LiveCC Streaming CLI')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: 0 for webcam, or RTSP URL')
    parser.add_argument('--model-path', type=str, 
                       default='chenjoya/LiveCC-7B-Instruct',
                       help='Path to LiveCC model')
    parser.add_argument('--query', type=str,
                       default='Please describe what you see.',
                       help='Query to ask about the video stream')
    parser.add_argument('--fps', type=int, default=2,
                       help='Target frames per second to capture')
    parser.add_argument('--max-pixels', type=int, default=384 * 28 * 28,
                       help='Maximum pixels for video processing')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file to save commentaries')
    parser.add_argument('--save-screenshots', action='store_true',
                       help='Save screenshot at start of each commentary segment')
    parser.add_argument('--screenshot-dir', type=str, default='screenshots',
                       help='Directory to save screenshots (default: screenshots)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Convert source to int if it's a number (for webcam)
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Initialize model
    print(f"Loading model: {args.model_path}")
    infer = LiveCCStreamingInfer(model_path=args.model_path)
    
    # Create frame buffer
    frame_buffer = LiveFrameBuffer(
        #buffer_size=1000,
        max_pixels=args.max_pixels,
        fps=args.fps
    )
    
    # Start capturing
    capture = LiveStreamCapture(source=source, target_fps=args.fps)
    capture.start(frame_buffer)
    
    try:
        print(f"Starting live commentary on stream: {source}")
        if args.save_screenshots:
            print(f"Screenshots will be saved to: {args.screenshot_dir}/")
        print("Press Ctrl+C to stop...\n")
        
        commentaries = live_cc_streaming(
            infer=infer,
            frame_buffer=frame_buffer,
            query=args.query,
            max_pixels=args.max_pixels,
            save_screenshots=args.save_screenshots,
            screenshot_dir=args.screenshot_dir,
        )
        
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        capture.stop()
        
        if args.output:
            result = {
                'source': str(source),
                'query': args.query,
                'commentaries': capture.buffer.commentaries if hasattr(capture.buffer, 'commentaries') else []
            }
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved results to {args.output}")


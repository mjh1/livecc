# LiveCC Streaming Video Guide

This guide explains how to use LiveCC with **live streaming video** instead of pre-recorded files.

## Overview

The streaming implementation allows you to:
- Process live webcam feeds
- Handle RTSP/RTMP streams
- Analyze real-time video from any OpenCV-compatible source
- Generate live commentary as the video streams

## Files

- `cli_streaming.py` - Main CLI for streaming video
- `infer_streaming.py` - Streaming inference implementation

## Quick Start

### 1. Webcam Stream

Process live video from your webcam (device 0):

```bash
python demo/cli_streaming.py --source 0
```

### 2. RTSP Stream

Process video from an RTSP stream:

```bash
python demo/cli_streaming.py --source rtsp://your-server:port/stream
```

### 3. Video File (for testing)

You can also test with a video file:

```bash
python demo/cli_streaming.py --source path/to/video.mp4
```

## Command-Line Options

```bash
python demo/cli_streaming.py \
    --source 0 \                          # Video source (0=webcam, URL for streams)
    --model-path chenjoya/LiveCC-7B-Instruct \  # Model to use
    --query "Describe what's happening" \  # Query for the model
    --fps 2 \                              # Frames per second to process
    --max-pixels 301056 \                  # Max pixels (384*28*28)
    --output results.json                  # Save results to JSON
```

### Parameters

- `--source`: Video input source
  - `0`, `1`, `2`, etc. for webcam device IDs
  - `rtsp://...` for RTSP streams
  - `rtmp://...` for RTMP streams
  - `http://...` for HTTP streams
  - File path for video files

- `--model-path`: HuggingFace model ID or local path
  - Default: `chenjoya/LiveCC-7B-Instruct`

- `--query`: Question to ask about the video
  - Default: `"Please describe what you see."`
  - Examples:
    - `"What objects do you see?"`
    - `"Describe the person's actions"`
    - `"What is happening in the scene?"`

- `--fps`: Target frames per second to capture and process
  - Default: `2` (recommended for real-time)
  - Lower values = less resource intensive
  - Higher values = more detailed analysis

- `--max-pixels`: Maximum pixel count for resizing
  - Default: `384 * 28 * 28 = 301,056`
  - Lower = faster, less detail
  - Higher = slower, more detail

- `--output`: Save commentary to JSON file (optional)

## How It Works

### Architecture

```
Video Source → LiveStreamCapture → LiveFrameBuffer → LiveCCStreamingInfer → Output
     (OpenCV)      (Thread)          (Deque)         (Model)            (Commentary)
```

1. **LiveStreamCapture**: Captures frames in a background thread
2. **LiveFrameBuffer**: Stores recent frames with timestamps
3. **LiveCCStreamingInfer**: Processes frames and generates commentary
4. **Output**: Prints commentary with timestamps

### Frame Processing

- Frames are captured at the specified FPS (default 2 fps)
- Recent frames are kept in a rolling buffer (default 100 frames)
- Model processes frames in batches with timestamps
- Commentary is generated for time intervals

### Latency

Expected latency depends on:
- Model size and device (GPU/CPU)
- Frame rate setting
- Network latency (for streams)
- Resolution settings

Typical latency: 1-3 seconds on GPU, 5-10 seconds on CPU

## Examples

### Example 1: Security Camera Feed

Monitor a security camera and describe activities:

```bash
python demo/cli_streaming.py \
    --source rtsp://admin:password@192.168.1.100:554/stream \
    --query "Describe any suspicious activities or unusual events" \
    --output security_log.json
```

### Example 2: Live Event Commentary

Generate commentary for a live event:

```bash
python demo/cli_streaming.py \
    --source 0 \
    --query "Provide commentary on what's happening" \
    --fps 3
```

### Example 3: Workshop Monitoring

Monitor a workshop or kitchen:

```bash
python demo/cli_streaming.py \
    --source rtsp://192.168.1.101/stream \
    --query "Describe the tools being used and actions being performed" \
    --max-pixels 200000  # Lower resolution for faster processing
```

## Troubleshooting

### Issue: "Cannot open video source"

**Solution**: 
- Check if webcam is connected and not in use by another app
- Verify RTSP/RTMP URL is correct and accessible
- Test the stream with VLC or another player first

### Issue: Very slow processing

**Solution**:
- Reduce `--fps` to 1 or lower
- Decrease `--max-pixels` to 150000 or less
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use a smaller model if available

### Issue: High memory usage

**Solution**:
- The frame buffer stores recent frames in memory
- Reduce FPS to capture fewer frames
- Modify `buffer_size` in `LiveFrameBuffer` class

### Issue: Network stream buffering

**Solution**:
- Add latency parameters to OpenCV VideoCapture
- Use lower resolution streams
- Check network bandwidth

## Advanced Usage

### Custom Frame Buffer Size

Edit `cli_streaming.py` to modify buffer size:

```python
frame_buffer = LiveFrameBuffer(
    max_pixels=args.max_pixels,
    fps=args.fps,
    buffer_size=200  # Increase buffer size (default: 100)
)
```

### Multiple Streams

To process multiple streams, create separate instances:

```python
from demo.infer_streaming import LiveCCStreamingInfer
from demo.cli_streaming import LiveFrameBuffer, LiveStreamCapture

# Create one model instance (shared)
infer = LiveCCStreamingInfer(model_path="chenjoya/LiveCC-7B-Instruct")

# Process multiple streams
streams = [
    ("rtsp://camera1/stream", "Camera 1"),
    ("rtsp://camera2/stream", "Camera 2"),
]

for source, name in streams:
    buffer = LiveFrameBuffer()
    capture = LiveStreamCapture(source)
    capture.start(buffer)
    # ... process each stream
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for 10-50x speedup
2. **Lower FPS**: 1-2 FPS is usually sufficient for commentary
3. **Optimize Resolution**: Balance between detail and speed
4. **Batch Processing**: Let the model process multiple frames together
5. **Use flash-attention**: Enabled automatically on CUDA devices

## Stopping the Stream

Press `Ctrl+C` to gracefully stop processing and save results (if `--output` is specified).

## Limitations

- Requires continuous network connection for stream sources
- Memory usage scales with buffer size and resolution
- Real-time processing depends on hardware capabilities
- Some stream formats may not be supported by OpenCV

## Next Steps

- See `demo/app.py` for Gradio web interface version
- Check `demo/cli.py` for file-based processing
- Read `README.md` for model details and training info


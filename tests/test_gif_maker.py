from PIL import Image
import os
from moviepy import ImageSequenceClip
import numpy as np

frame_dir = 'results/overcooked/gifs/frames'

# Now load the PNGs and create the video
print("\nLoading PNGs and creating video...")
frames = []
scale_factor = 4  # Increase resolution by 4x

for i in range(50):
    # Use 3-digit zero-padding for all frames to match the actual filenames
    frame_path = os.path.join(frame_dir, f'frame_{i:03d}.png')
    if os.path.exists(frame_path):
        # Read image and convert to numpy array
        img = Image.open(frame_path)
        # Scale up the image using high-quality resampling
        img = img.resize((img.width * scale_factor, img.height * scale_factor), Image.Resampling.LANCZOS)
        frames.append(np.array(img))
        print(f"Loaded frame {i} - Shape: {frames[-1].shape}")
    else:
        print(f"Warning: Frame {i} PNG not found at {frame_path}!")

print(f"\nLoaded {len(frames)} frames from PNGs")

# Create and save the video
output_path = 'results/overcooked/gifs/frame-load-test.mp4'
print("\nCreating video...")
print(f"Number of frames to write: {len(frames)}")

# Create video clip (2 fps = 500ms per frame)
clip = ImageSequenceClip(frames, fps=2)
# Use high quality settings for video encoding
clip.write_videofile(
    output_path, 
    fps=2, 
    codec='libx264',
    bitrate='8000k',  # High bitrate for better quality
    preset='slow',    # Slower encoding for better quality
    threads=4         # Use multiple threads for faster encoding
)

print(f"\nVideo saved successfully to {output_path}!")

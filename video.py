import cv2
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--image-dir",
    type=Path,
    required=True,
    help="Path to images sequence",
)
args = parser.parse_args()

output_file = "output.mp4"
fps = 30

images = sorted(args.image_dir.glob("*.png"), key=lambda x: int(x.stem.rsplit("-", 1)[-1]))

if not images:
    raise ValueError("No images found")

# Read first image to determine video size
first = cv2.imread(str(images[0]))
height, width = first.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(
    output_file,
    fourcc,
    fps,
    (width, height),
)

for image_path in images:
    frame = cv2.imread(str(image_path))

    if frame is None:
        print(f"Skipping: {image_path}")
        continue

    # Make sure all images have the same dimensions
    frame = cv2.resize(frame, (width, height))
    video.write(frame)

video.release()

print(f"Created: {output_file}")

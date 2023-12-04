import cv2
import sys
import os

import imageio
import os

# Path to the directory containing JPG files
input_directory = sys.argv[1]

# Output video file name
output_video = 'output.mp4'
ref = {}

# List all JPG files in the directory
image_files = [file for file in os.listdir(input_directory) if file.endswith('.jpg')]
for i in range(len(image_files)):
    str = image_files[i]
    numbers = str[15:]
    # print(numbers[:-4])
    ref[int(numbers[:-4])] = image_files[i]

image_files.sort()
# Create an ImageIO writer with FFmpeg
writer = imageio.get_writer(output_video, fps=30)

# Iterate through image files and add them to the video
for i in range(len(image_files)):
    image_path = os.path.join(input_directory, ref[i])
    img = imageio.imread(image_path)
    writer.append_data(img)

# Close the writer
writer.close()

print(f"Video created: {output_video}")
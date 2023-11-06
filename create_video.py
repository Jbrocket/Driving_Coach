import cv2
import sys
import os

if __name__ == "__main__":
    frame_width = 1392  # Adjust to your frame width
    frame_height = 512  # Adjust to your frame height
    frame_rate = 10  # Adjust to your desired frame rate (frames per second)

    frame_directory = sys.argv[1]

    frame_files = [f for f in os.listdir(frame_directory) if f.endswith('.png')]
    frame_files.sort()

    # Create the VideoWriter object
    out = cv2.VideoWriter('input.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height), isColor=True)

    # Loop through the image files and write them to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frame_directory, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the VideoWriter and close the video file
    out.release()
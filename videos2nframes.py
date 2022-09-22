# Extract frames from .mp4 videos into a folder with the name of each file

import cv2
import os
import random

def video2nframes(video_path, output_path, n, video_name):
    # video_path: path to video file
    # output_path: path to output directory
    # n: number of frames to extract

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Random list of n frames
    frames_index = random.sample(range(1, num_frames), n)

    # Save n frames
    count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if count in frames_index:
                image_path = os.path.join(output_path, video_name + "-%d.jpg" % count)
                cv2.imwrite(image_path, frame)
                print(image_path + " saved")
            count += 1
        else:
            break
    

try:
    videos_path = os.path.abspath(input("Enter the path of the folder containing the videos: "))
except:
    print("The path does not exist")
    exit()


# Create a folder for frames in the same directory as the videos
os.chdir(videos_path)
if not os.path.exists('frames'):
    os.makedirs('frames')
output_path = os.path.join(videos_path, 'frames')


# Extract frames from each video in the current directory
for path in os.listdir(videos_path):
    if path.endswith('.mp4'):
        video_name = path.split(".")[0]

        # Create a folder for each video with the name of the video
        path_video = os.path.join(output_path, video_name)
        if not os.path.exists(path_video):
            os.makedirs(os.path.join(output_path, video_name))

        # Extract frames and save n images them in the folder of the video
        n = 10
        video2nframes(path, path_video, n, video_name)

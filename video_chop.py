import os
import sys
import argparse
import cv2
from tqdm import tqdm
from pathlib import Path

def chop_video(video_path: str, L: int) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    total_frames = (total_frames // L) * L

    video_frames = []
    for _ in tqdm(range(total_frames), desc='Reading video frames'):
        ret, frame = video.read()
        if ret:
            video_frames.append(frame)

    # Calculate the maximum depth level
    max_depth = 0
    while L ** (max_depth) <= total_frames // L:
        max_depth += 1

    dir_name = Path(video_path).stem

    for curr_depth in range(max_depth):
        num_splits = L ** curr_depth
        frames_per_split = total_frames // num_splits
        if dir_name == "":
            dir_name = f"depth_{curr_depth}"
        else:
            dir_name = os.path.join(dir_name, f"depth_{curr_depth}")
        os.makedirs(dir_name, exist_ok=True)

        for i in tqdm(range(num_splits), desc=f'Depth {curr_depth}'):

            os.makedirs(os.path.join(dir_name, f"part_{i//L}"), exist_ok=True)
            output_filename = f"{dir_name}/part_{i//L}/subset_{i%L}.mp4"
            height, width, _ = video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

            start_index = i * frames_per_split
            end_index = (i + 1) * frames_per_split

            for j in tqdm(range(start_index, end_index), desc=f'Subset {i}, {len(range(start_index, end_index))} Frames'):
                out.write(video_frames[j])

            out.release()
            # create a txt file alongside the video
            with open(f"{dir_name}/part_{i//L}/subset_{i%L}.txt", "w") as f:
                f.write(f"")

    video.release()

def main():
    parser = argparse.ArgumentParser(description="Chop a video file into subsets of frames.")
    parser.add_argument("video_file", help="Path to the video file.")
    parser.add_argument("--L", help="Num of splits on each level.")
    args = parser.parse_args()
    chop_video(args.video_file, int(args.L))

if __name__ == "__main__":
    main()

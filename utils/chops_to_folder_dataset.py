import os
import sys
import argparse
import cv2
from tqdm import tqdm
import shutil
from pathlib import Path

def write_as_video(output_filename, video_frames, overwrite_dims, width, height, fps):
    if overwrite_dims:
        height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    for j in video_frames:
        out.write(j)

    out.release()

def read_first_frame(video_path):
    patience = 5
    p = 0
    video = cv2.VideoCapture(video_path)
    ret = False
    while not ret:
        ret, frame = video.read()
        p += 1
        if p > patience:
            raise Exception(f'Cannot read the video at {video_path}')
    video.release()
    return frame

def get_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    return fps   

def move_the_files(init_path, L, depth, overwrite_dims, width, height, overwrite_fps, fps):

    folder_dataset_path = os.path.join(init_path, 'folder_dataset')
    depth_name = init_path

    t_counter=0
    for d in range(0, depth):
        for j in range(L**(d-1) if d > 1 else 1):
            for i in range(L if d > 0 else 1):
                t_counter+=1
    tq = tqdm(total=t_counter)

    for d in range(0, depth):
        depth_name = os.path.join(depth_name, f'depth_{d}')
        for j in range(L**(d-1) if d > 1 else 1):
            part_path = os.path.join(depth_name, f'part_{j}')
                # sample the text info for the next subset
            for i in range(L if d > 0 else 1):
                txt_path = os.path.join(part_path, f'subset_{i}.txt')
                
                # go to the subset for video frames sampling
                next_depth_name = os.path.join(depth_name, f'depth_{d+1}')
                next_part_path = os.path.join(next_depth_name, f'part_{i}') # `i` cause we want to sample each corresponding *subset*

                # depths > 0 are *guaranteed* to have L videos in their part_j folders
                
                # now sampling each first frame at the next level
                L_frames = [read_first_frame(os.path.join(next_part_path, f'subset_{k}.mp4')) for k in range(L)]
                
                # write all the L sampled frames to an mp4 in the folder dataset
                if overwrite_fps:
                    fps = get_fps(os.path.join(next_part_path, f'subset_{0}.mp4'))
                
                write_as_video(os.path.join(folder_dataset_path, f'depth_{d}_part_{j}_subset{i}.mp4'), L_frames, overwrite_dims, width, height, fps)
                shutil.copy(txt_path, os.path.join(folder_dataset_path, f'depth_{d}_part_{j}_subset{i}.txt'))

                t += 1
                tq.set_description(f'Depth {d}, part {j}, subset{i}')
                tq.update(t)
    
    tq.close() 

def main():
    parser = argparse.ArgumentParser(description="Convert the chopped labeled tree-like data into a FolderDataset")
    parser.add_argument("video_file", help="Path to the video file.")
    parser.add_argument("--L", help="Num of splits on each level.")
    parser.add_argument("--D", help="Tree depth")
    parser.add_argument("--overwrite_dims", help="Preserve the original video dims", action="store_true")
    parser.add_argument("--w", help="Output video width", default=384)
    parser.add_argument("--h", help="Output video height", default=256)
    parser.add_argument("--overwrite_fps", help="Preserve the original video fps", action="store_true")
    parser.add_argument("--fps", help="Output video fps", default=12)
    args = parser.parse_args()
    move_the_files(args.video_file, int(args.L), int(args.D))

if __name__ == "__main__":
    main()
    
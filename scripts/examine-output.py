import glob
from matplotlib import pyplot as plt
import os
import imageio

# Update the output folder before use
# output_folder = '/home/xinzhi/Text-To-Video-Finetuning/output/unique_token_dog_only_1700608471/'
output_folder = '/Users/masoudcharkhabi/github/Text-To-Video-Finetuning/output/'

def ExtractTitle(file_path):
    file_name = os.path.basename(file_path)
    prompt = file_name.split('_')[0]
    iterations = file_name.split('_')[-1].split(' ')[0]
    title = "Prompt: " + prompt + " rendering at " + iterations + " iterations."
    return title

def ExtractFrames(file_path, sampling_rate = 1):
    reader = imageio.get_reader(file_path)
    frames = []
    for i, frame in enumerate(reader):
        if i % sampling_rate == 0:
            frames.append(frame)
    return frames

video_files = glob.glob(output_folder + '*')
video_files.sort()
for file_path in video_files: 
    title = ExtractTitle(file_path)
    frames = ExtractFrames(file_path, sampling_rate=50)
    for frame in frames:
        plt.imshow(frame)
        plt.title(title)
        plt.show()

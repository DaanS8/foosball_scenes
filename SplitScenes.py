from __future__ import print_function, division

import math

import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import subprocess
import json
import cv2

if __name__ == '__main__':
    with torch.no_grad():
        for video_file in os.listdir():
            if len(video_file) > 4 and video_file[-3:] == 'mp4':
                    print(video_file)
                    # Aks for path to video
                    path_exists = True
                    # video_file = 'test_'+str(_nb)+'.mp4'
                    while not path_exists:
                        video_file = input('Path to video to extract frames from: ')
                        if os.path.isfile(video_file) and video_file[-3:] == 'mp4':
                            path_exists = True
                        else:
                            print('Given path is not a valid mp4 file, try again.')

                    # Initialising pytorch parameters
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(0.3413, 0.2718)
                    ])

                    model = models.resnet18()
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, 3)
                    model.load_state_dict(torch.load('epoch_' + str(1) + '.model'))

                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    model_ft = model.to(device)

                    model.eval()

                    frames = []  # we are going to save the frames here.


                    def get_frames_metadata(file):
                        command = '"{ffexec}" -hide_banner -loglevel error -select_streams v -show_frames -print_format json "{filename}"'.format(ffexec='ffprobe', filename=file)
                        response_json = subprocess.check_output(command, shell=True, stderr=None)
                        frames = json.loads(response_json)["frames"]
                        iframes = []
                        time_stamps = []
                        counter = 0
                        prev_size = -1
                        alpha = 0.2
                        first = True
                        for frame in frames:
                            if frame["media_type"] == "video":
                                cur_size = int(frame["pkt_size"])
                                if first:
                                    first = False
                                    prev_size = cur_size
                                time_stamps.append(float(frame["pts_time"]))
                                if frame["pict_type"] == "I":
                                    iframes.append(counter)
                                elif prev_size * 5 < cur_size:
                                    # too large to be a similar consecutive frame
                                    # this a kind of iframe
                                    iframes.append(counter)
                                counter += 1
                                prev_size = (1 - alpha) * prev_size + alpha * cur_size
                        return iframes, time_stamps

                    def get_frame_guess(video, frame_nb):
                        video.set(1, frame_nb)
                        succes, frame = video.read()
                        h, w, c = frame.shape
                        dim = (224, int(224 * h / w))
                        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                        return torch.max(model(transform(frame).unsqueeze(dim=0).to(device)).to('cpu'), 1)[1]

                    iframes, time_stamps = get_frames_metadata(video_file)
                    print(iframes)

                    video = cv2.VideoCapture(video_file)
                    low_frame_nb = iframes.pop(0)
                    low_frame_class = get_frame_guess(video, low_frame_nb)

                    class_count = {0: 0,
                                   1: 0,
                                   2: 0}

                    os.mkdir('out/' + video_file[:-4])

                    tmp_counter = 0
                    for frame_nb in iframes:
                        cur_class = get_frame_guess(video, frame_nb)
                        if cur_class != low_frame_class:
                            if get_frame_guess(video, frame_nb-1) == low_frame_class:
                                # prev frame is part of previous segment
                                nb_frames = frame_nb - low_frame_nb
                                if nb_frames > 5:
                                    time_stamp = time_stamps[frame_nb-1] - time_stamps[low_frame_nb]
                                    print(nb_frames, time_stamp)
                                    out_file = 'out/' + video_file[:-4] + '/' + str(int(low_frame_class)) + '_' + str(class_count[int(low_frame_class)]) + '.mp4'
                                    command = 'ffmpeg -hide_banner -loglevel error -an -sn -dn -ss {seek} -t {time} -accurate_seek -i {in_file} -c copy {out}'.format(
                                        seek=str(time_stamps[low_frame_nb]), time=str(time_stamp), in_file=video_file, out=out_file)
                                    class_count[int(low_frame_class)] += 1
                                    subprocess.call(command, shell=True)
                                else:
                                    print(f'Ignored {nb_frames} at frame {frame_nb}.')
                                low_frame_nb = frame_nb
                                low_frame_class = cur_class

                            else:
                                upper_bound = frame_nb - 1
                                guess_upperbound = cur_class
                                prev_guess = upper_bound
                                cur_guess_frame = low_frame_nb + math.ceil((upper_bound - low_frame_nb) / 2)
                                while cur_guess_frame != upper_bound:
                                    guess = get_frame_guess(video, cur_guess_frame)
                                    if low_frame_class != guess:
                                        upper_bound = cur_guess_frame
                                        guess_upperbound = guess
                                        new_guess = cur_guess_frame - math.ceil((abs(prev_guess - cur_guess_frame)) / 2)
                                    else:
                                        new_guess = cur_guess_frame + math.ceil((abs(prev_guess - cur_guess_frame)) / 2)
                                    prev_guess = cur_guess_frame
                                    cur_guess_frame = new_guess
                                nb_frames = upper_bound - low_frame_nb
                                if nb_frames > 5:
                                    time_stamp = time_stamps[upper_bound-1] - time_stamps[low_frame_nb]
                                    print(tmp_counter, 'nb_frames', nb_frames, 'duration', time_stamp)
                                    out_file = 'out/' + video_file[:-4] + '/' + str(int(low_frame_class)) + '_' + str(
                                        class_count[int(low_frame_class)]) + '.mp4'
                                    command = 'ffmpeg -hide_banner -loglevel error -an -sn -dn -ss {seek} -t {time} -accurate_seek -i {in_file} -c copy {out}'.format(
                                        seek=str(time_stamps[low_frame_nb]), time=str(time_stamp), in_file=video_file, out=out_file)
                                    class_count[int(low_frame_class)] += 1
                                    subprocess.call(command, shell=True)
                                else:
                                    print(f'Ignored {nb_frames} at frame {frame_nb}.')
                                low_frame_nb = upper_bound
                                low_frame_class = cur_class
                            tmp_counter += 1

                    time_stamp = time_stamps[-1] - time_stamps[low_frame_nb]
                    if time_stamp > 5/30:
                        out_file = 'out/' + video_file[:-4] + '/' + str(int(low_frame_class)) + '_' + str(
                            class_count[int(low_frame_class)]) + '.mp4'
                        command = 'ffmpeg -an -sn -dn -ss {seek} -t {time} -accurate_seek -i {in_file} -c copy {out}'.format(
                            seek=str(time_stamps[low_frame_nb]), time=str(time_stamp), in_file=video_file, out=out_file)


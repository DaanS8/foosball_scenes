from __future__ import print_function, division

import math

import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import subprocess
import json
import cv2

""""
This script can be used to visualise the SplitScene extraction.

It can be used to
 - Visually check if the scenes are correctly detected.
 - Check which frames are examined for the SplitScene extraction.
 - Debug why a specific segment is wrongly split.

The resulting video shows:
- Certainty of prediction of current frame
- Class of the current segment
    - Green: matches the prediction of the exact frame.
    - Red: is different than the prediction of this frame.
- Class of the current frame
    - White: normal frame
    - Blue: iframe (anchor frame)
    - Light Blue: this frame contains a lot of information

The resulting 'video' pauses for half a second if:
1) The class segment is different from the frame class
2) The frame is either an iframe or a frame with a lot of information
"""


video_file = 'archive/test_6.mp4'


def get_frames_metadata(file):
    """
    Get metadata of all frames, and search for iframes (aka anchor frames), or frames containing a lot of information.
    By only evaluating these frames, speedup is achieved compared to needing to evaluate every frame.
    """
    # command takes up 1/3 of total runtime, TODO improve?
    command = 'ffprobe -hide_banner -loglevel error -select_streams v -show_frames -print_format json "{filename}"'.format(filename=file)
    response_json = subprocess.check_output(command, shell=True, stderr=None)
    frames = json.loads(response_json)["frames"]

    important_frames = []  # all frames to evaluate
    counter = 0

    prev_size = -1
    first = True
    alpha = 0.2  # constant for exponential moving average

    counter_true_iframes = 0
    counter_lots_of_info = 0
    lots_of_info = []  # used for display purposes

    for video_frame in frames:
        # frame size ∼ nb bits needed for that frame ∼ information contained in frame
        cur_size = int(video_frame["pkt_size"])

        if first:  # initialise prev_size
            first = False
            prev_size = cur_size

        if video_frame["pict_type"] == "I":
            # iframe!
            important_frames.append(counter)
            counter_true_iframes += 1

        elif prev_size * 5 < cur_size:
            # too large to be a normal consecutive frame
            # treat as anchor frame
            important_frames.append(counter)
            lots_of_info.append(counter)
            counter_lots_of_info += 1

        # use exponential weighted average as estimate of a normal frame size
        prev_size = (1 - alpha) * prev_size + alpha * cur_size
        counter += 1
    print('Frames in video: {total}. '
          'Frames to evaluate: {to_examine_count}. '
          'Nb of iframes: {iframes_count}. '
          'Nb of frames that contain a lot of information: {info_counter}.'.format(
            total=counter, to_examine_count=len(important_frames), iframes_count=counter_true_iframes, info_counter=counter_lots_of_info))
    return important_frames, lots_of_info


if __name__ == '__main__':
    with torch.no_grad():  # autograd slows down calculations
        # Initialising pytorch parameters
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.3413, 0.2718)
        ])

        # Use resnet 18 model, but with only 3 classes
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)

        # Loading (best) trained model
        model.load_state_dict(torch.load('model_1.pt'))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # try GPU
        model_ft = model.to(device)

        model.eval()

        iframes, abnormal = get_frames_metadata(video_file)

        video = cv2.VideoCapture(video_file)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # weird location for function to not pass model, video, transform and device as parameters
        def get_frame_guess(frame_nb):
            """
                Get class evaluation of model of the given frame.

                Set video at correct frame, get frame and transform, forward pass through model,
                select largest output as class.
            """
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
            _, frame = video.read()
            h, w, c = frame.shape
            dim = (224, int(224 * h / w))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            return torch.max(model(transform(frame).unsqueeze(dim=0).to(device)).to('cpu'), 1)[1]

        # setup for segment detection
        low_frame_nb = iframes[0]
        low_frame_class = get_frame_guess(low_frame_nb)

        classes = []  # list to keep track of the classes of a frame of the detected segments
        for frame_nb in iframes[1:]:
            cur_class = get_frame_guess(frame_nb)
            if cur_class != low_frame_class:
                if get_frame_guess(frame_nb - 1) == low_frame_class:
                    # prev frame is part of previous segment
                    nb_frames = frame_nb - low_frame_nb
                    for q in range(nb_frames):
                        classes.append(low_frame_class.item())
                    low_frame_nb = frame_nb
                    low_frame_class = cur_class
                else:
                    # upper_bound = lowest frame_nb of cur_class
                    upper_bound = frame_nb - 1
                    guess_upperbound = cur_class
                    prev_guess = upper_bound
                    cur_guess_frame = low_frame_nb + math.ceil((upper_bound - low_frame_nb) / 2)
                    while cur_guess_frame != upper_bound:
                        guess = get_frame_guess(cur_guess_frame)
                        if low_frame_class != guess:
                            upper_bound = cur_guess_frame
                            guess_upperbound = guess
                            new_guess = cur_guess_frame - math.ceil((abs(prev_guess - cur_guess_frame)) / 2)
                        else:
                            new_guess = cur_guess_frame + math.ceil((abs(prev_guess - cur_guess_frame)) / 2)
                        prev_guess = cur_guess_frame
                        cur_guess_frame = new_guess
                    nb_frames = upper_bound - low_frame_nb
                    for q in range(nb_frames):
                        classes.append(low_frame_class.item())
                    low_frame_nb = upper_bound
                    low_frame_class = cur_class
        for i in range(length - low_frame_nb):
            classes.append(low_frame_class.item())

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(length):
            print(i)
            success, frame = video.read()
            h, w, c = frame.shape
            dim = (224, int(224 * h / w))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            model_out = model(transform(frame).unsqueeze(dim=0).to(device)).to('cpu')
            exp = torch.nn.functional.normalize(torch.exp(model_out))[0]
            guess = torch.max(model_out, 1)[1].item()
            perc = exp[guess].item()
            guess = str(guess)

            font_color = (255, 255, 255)
            if i in iframes:
                font_color = (255, 0, 0)
                if i in abnormal:
                    font_color = (255, 150, 105)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 3
            font_thickness = 2
            x, y = 125, 120
            frame = cv2.putText(frame, guess, (x, y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

            font_color = (0, 255, 0)
            if str(classes[i]) != guess:
                font_color = (0, 0, 255)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 3
            font_thickness = 2
            x, y = 50, 120
            frame = cv2.putText(frame, str(classes[i]), (x, y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1
            font_thickness = 1
            x, y = 50, 50
            frame = cv2.putText(frame, str("{:0.2f}%".format(perc * 100)), (x, y), font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)

            wait = 0
            if not i in iframes and str(classes[i]) == guess:
                wait = int(10 * math.exp((1 - perc) * 8))

            cv2.imshow('vid', frame)
            cv2.waitKey(wait)








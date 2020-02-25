
from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display
import glob
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

directory_path = './data'
mp4_path = './deepfake-detection-challenge/train_sample_videos/*.mp4'


if not os.path.exists(directory_path):
    os.mkdir(directory_path)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)



if __name__ == '__main__':

    filelists = glob.glob(mp4_path) 

    print('filelists: ', len(filelists))
    for path in filelists:
        
        folder_path = os.path.join(directory_path, path.split('/')[-1])
        print('folder_path: ', folder_path)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        video = mmcv.VideoReader(path)
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
        for i, frame in enumerate(frames):
            print('\rTracking frame: {}'.format(i + 1), end='')
            # Detect faces
            boxes, _ = mtcnn.detect(frame)
            cropped_images = []
            if boxes is None: continue
            for box in boxes:
                crop = frame.crop(box.tolist()) # check 하기
                cropped_images.append(crop)
            # save
            for j, img in enumerate(cropped_images):
                img.save(os.path.join(folder_path, '{}-f-{}.jpg'.format(i, j)))
        
    print('\nDone')

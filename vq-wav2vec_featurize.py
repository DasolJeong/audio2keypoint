import argparse
import glob
import imp
import os, cv2
import os.path as osp
from os.path import isfile, join
import audio

import soundfile as sf
import torch
import fairseq
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


try:
    import tqdm
except:
    print("Install tqdm to use --log-format=tqdm")



class FilesDataset:
    def __init__(self, files_path):
        self.path_list = files_path                     # lrs2/
        audio_image_list = glob.glob(self.path_list + '*')  # *: 553534534034, 5551635435444, ...

        self.dataset_audio = []
        self.dataset_image = []


        for audio_image_path in audio_image_list:
            for audio_path in glob.glob(audio_image_path + '/**/*.wav'):    # lrs2/553534534034/00001/audio.wav
                audio_name = audio_path.split('/')[-1].replace('.wav', '')
                self.dataset_audio.append([audio_path, audio_name])         # [lrs2/553534534034/00001/audio.wav, audio]
            image_frames = []
            # for frame_id in range(len(glob.glob(audio_image_path + ' /**/*.jpg'))):
            #     frame = join(audio_image_path, '{}.jpg'.format(frame_id))

            for image_path in glob.glob(audio_image_path + '/**/*.jpg'):    # lrs2/553534534034/00001/000.jpg
                if not isfile(image_path):
                    return None
                image_frames.append(image_path)
                # image_name.append(image_path.split('/')[-1])
                self.dataset_image.append([image_path, image_frames])

        self.all_videos = glob.glob(self.path_list + '**/*')            # lrs2/553534534034/00001/



    def read_frames(self, image_frames):
        if image_frames is None: return None
        window = []
        for fname in image_frames:
            img = cv2.imread(fname)
            if img is None:
                return None

            window.append(img)
        
        return window

    def prepare_frames(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x
        
    
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, index):
        vidname = self.all_videos[index]

        wavpath = join(vidname, "audio.wav")
        wav = audio.load_wav(wavpath, 16000)

        frames = []
        for frame in glob.glob(vidname + '/*.jpg'):
            if not isfile(frame):
                return None
            frames.append(frame)
        imgs = self.read_frames(frames)
        imgs = self.prepare_frames(imgs)
        img = torch.FloatTensor(imgs)

        wav = torch.FloatTensor(wav)

        return wav, img


DATA_PATH = './lrs2_preprocessed2/'

dataset = FilesDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=32, num_workers=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")





wav_input = tokenizer()


import argparse
import sys
import os
import numpy as np
import audio
from PIL import Image
from skimage import io
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pickle
import time
import datasets
import torchaudio
import audio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor, AutoFeatureExtractor
# from transformer_DS import DSNet
from matplotlib import pyplot as plt
from torchvision import datasets, transforms, utils

from tqdm import tqdm

from scheduler import CycleScheduler
import distributed as dist
import cv2

# from dataset102 import Facial_Dataset
from dataLoader import LRS2_Dataset
import face_alignment


device = torch.device("cuda:0")

class DSNet(nn.Module):
    "train model"
    def __init__(self, channel):
        super(DSNet, self).__init__()
        # audio_output = audio_output.transpose(1, 2)  # [b, 784, 127]
        # audio_output = [1, 66, 512] -> Conv1d(66, 33, 1, …) -> [1, 33, 512]
        self.linear1 = nn.Conv1d(in_channels=channel + 1, out_channels=channel//2, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv1d(512, 64, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, r, x):
        x_r = torch.cat(r, x)
        x_r = self.linear1(x_r)
        x_t = x.permute(0, 2, 1)
        x_t = self.linear2(x_t)
        # x = self.sigmoid(x)
        # return np.array(x.cpu())
        return x_t

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(68*2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 68*2),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noise = torch.tensor(noise).to(device)
    noisy_img = img + noise
    return noisy_img



def train(epoch, loader, model, wav2vec2_processor, wav2vec2_model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()
    

    mse_sum = 0
    mse_n = 0

    model.train()

    model.zero_grad()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda')

    all_loss = 0.0
    #vae_loss = 0.0
    wav2vec_loss = 0.0
    cnt2 = 0

    for i, (img_list, audio_path) in enumerate(loader):
        vae_loss = 0.0
        cnt = 0
        t1 = time.time()

        waveform = audio.load_wav(audio_path[0], sr=16000)        
        waveform = torch.from_numpy(waveform)
        waveform = waveform.to(device)
        input_value = wav2vec2_processor(waveform, return_tensors='pt', sampling_rate=16000).input_values
        # print(input_value)
        input_value = input_value.to(device)
        output = wav2vec2_model(input_value).extract_features
        # print('output:', output.shape)

        len_list = output.size(1)
        len_img = len(img_list)
        all_encoded = torch.zeros((len_img, 64))
        refer = []
        for t, fname in enumerate(img_list):
            optimizer.zero_grad()
            # image = Image.open(fname[0]).convert('RGB')
            try:
                image = io.imread(fname[0])
                preds = face_aligner.get_landmarks(image)[-1]

            except:
                break

            else:
                cnt += 1

                preds_tensor = torch.tensor(preds).to(device)
                # print(preds_tensor.shape)   # ([68, 2])
                noisy_x = add_noise(preds_tensor)  # 입력에 노이즈 더하기
                noisy_x = noisy_x.view(-1, 68*2)
                y = preds_tensor.view(1, 136)
                # label = label.to(device)
                encoded, decoded = model(noisy_x)
                # print(encoded.shape)    # ([1, 64])
                # print(decoded.shape)    # ([1, 136])
                all_encoded[t] = encoded
                if t == 0:
                    refer = encoded
                loss = criterion(y, decoded)
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                vae_loss += loss.item()
                t2 = time.time()
            vae_loss_now = vae_loss / cnt
            
        all_encoded = all_encoded.unsqueeze(-1).to(device)

        refer = all_encoded[0]
        #print("all_encoded:", all_encoded.shape)
        dsnet = DSNet(channel=len_list).to(device)
        cnt2 += 1
        ds_result = dsnet(refer, output_c)

        #print("ds_result: ", ds_result.shape)
        #optimizer_W = torch.optim.Adam(dsnet.parameters(), lr=0.0001)
        loss2 = criterion(ds_result, all_encoded)
        wav2vec_loss += loss2.item()
        wav2vec_loss_now = wav2vec_loss / cnt2
        loss2.backward()
        optimizer.step()
        wav2vec_loss += loss2.item()

        #######all_loss = vae_loss + wav2vec_loss
        #all_loss.backward()
        #optimizer.step()


        if scheduler is not None:
            scheduler.step()
        optimizer.step()

#        comm = dist.all_gather(comm)

 #       for part in comm:
  #          mse_sum += part["mse_sum"]
   #         mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; loss: {vae_loss_now:.4f};"
                    f"latent: {loss.item():.3f}; "
                    f"lr: {lr:.5f}"
                )
            )
        
            if i % 100 == 0:
                model.eval()

                # waveform = audio.load_wav(audio_path[0][0], sr=16000)
                waveform = audio.load_wav(audio_path[0], sr=16000) 
                input_value = wav2vec2_processor(waveform, return_tensors='pt').input_values
                input_value = input_value.to(device)
                audio_output = wav2vec2_model(input_value).extract_features
                # audio_output = audio_output.transpose(1, 2)


                with torch.no_grad():
                    out = dsnet(audio_output)
                    # print(out.shape)
                    out = out.permute(0, 2, 1)
                    out = model.decoder(out)
                    out = out.reshape([1, 68, 2, -1])
                    print(out[:,:,:,1])
                    # sample = vqvae.decoder()


def main(args):
    device = torch.device("cuda:0")
    
    args.distributed = dist.get_world_size() > 1

    dataset = LRS2_Dataset(args.path)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=1 // args.n_gpu, sampler=sampler, num_workers=2
    )

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load('checkpoint/audio2landmark_010.pt'))
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    wav2vec2_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device)
    # autoencoder = Autoencoder().to(device)
    criterion = nn.MSELoss()
    # optimizer_W = torch.optim.Adam(dsnet.parameters(), lr=0.0001)
    model = DSNet().to(device)


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )



    for i in range(args.epoch):
        SAVE_PATH = f"checkpoint/audio2landmark_wav2vaec2_{str(i + 1).zfill(3)}.pt"

        train(i, loader, autoencoder, wav2vec2_processor, wav2vec2_model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save({
                "model_state_dict": model.state_dict(),
            }, SAVE_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    # parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:10002")

    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--path", type=str, default='./lrs2_preprocessed/')

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))

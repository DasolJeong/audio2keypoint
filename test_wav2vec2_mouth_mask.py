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

from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
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
    def __init__(self):
        super(DSNet, self).__init__()
        
        self.linear_seq = nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        # audio_output 3 channel = [1, 3, 512] -> [1, 1, 512]
    
        self.linear1 = nn.Conv1d(608, 512, kernel_size=3, stride=1, padding=1)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv1d(128, 136, kernel_size=3, stride=1, padding=1)


    def forward(self, len_img, ref, x):
        all_seq = torch.zeros((len_img, 136, 1)).to(device)
        splits = x.split(2, dim=1)    # ref: [len, 96, 1] x: [1, len, 512] splits: [1, 2, 512]

        for i in range(len_img):
            sizes = splits[i].size()
            # splits = x.split(2, dim=1)    # ref: [len, 96, 1] x: [1, len, 512] splits: [1, 2, 512]
            if sizes[1] == 2:
                s = self.linear_seq(splits[i])
                s = s.permute(0, 2, 1)
                r = ref[i, :, :].unsqueeze(0)
                c = torch.cat([r, s], dim=1)
                c = self.linear1(c)
                c = self.ReLU(c)
                c = self.linear2(c)
                c = self.ReLU(c)
                c = self.linear3(c)
                c = self.ReLU(c)
                c = self.linear4(c)

                all_seq[i] = c


        return all_seq



def generate_landmarks(frames_list, face_aligner):

    frame_landmark_list = []
    fa = face_aligner
    
    for i in range(len(frames_list)):
        try:
            input = frames_list[i]
            preds = fa.get_landmarks(input)[0]

            dpi = 100
            fig = plt.figure(figsize=(input.shape[1]/dpi, input.shape[0]/dpi), dpi = dpi)
            ax = fig.add_subplot(1,1,1)
            ax.imshow(np.ones(input.shape))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            #chin
            ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
            #left and right eyebrow
            ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            #nose
            ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            #left and right eye
            ax.plot(preds[36:42,0],preds[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            ax.plot(preds[42:48,0],preds[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            #outer and inner lip
            ax.plot(preds[48:60,0],preds[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
            ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
            ax.axis('off')

            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frame_landmark_list.append((input, data))
            plt.close(fig)
        except:
            print('Error: Video corrupted or no landmarks visible')
    
    for i in range(len(frames_list) - len(frame_landmark_list)):
        #filling frame_landmark_list in case of error
        frame_landmark_list.append(frame_landmark_list[i])
    
    
    return frame_landmark_list


def test(epoch, loader, wav2vec2_processor, wav2vec2_model, dsnet, optimizerB, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()
    
    dsnet.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda')

    wav2vec_loss = 0.0
    cnt2 = 0

    for i, (img_list, audio_path) in enumerate(loader):
        
        t1 = time.time()

        waveform = audio.load_wav(audio_path[0], sr=16000)        
        waveform = torch.from_numpy(waveform)
        waveform = waveform.to(device)

        input_value = wav2vec2_processor(waveform, return_tensors='pt', sampling_rate=16000).input_values
        # print(input_value)
        input_value = input_value.to(device)
        output = wav2vec2_model(input_value).extract_features
        output = output.to(device)
        # print('output:', output.shape)

        len_list = output.size(1)
        len_img = len(img_list)
        # all_encoded = torch.zeros((len_img, 64))
        all_preds = torch.zeros((len_img, 136, 1)).to(device)
        refer = torch.zeros((len_img, 96, 1)).to(device)
        # refer = []
        for t, fname in enumerate(img_list):
            # optimizer.zero_grad()
            # image = Image.open(fname[0]).convert('RGB')
            try:
                image = io.imread(fname[0])
                preds = face_aligner.get_landmarks(image)[-1]
                preds = preds / 160

            except:
                break

            else:
                # cnt += 1
                preds_tensor = torch.tensor(preds).to(device)
                y = preds_tensor.view(1, 136)
                y = y.unsqueeze(-1).to(device)
                all_preds[t] = y
                refer[t] = y[:, :96, :]
                t2 = time.time()
            if isinstance(refer, list) == True:
                break

        with torch.no_grad():
                    all_result = all_preds

                    out = dsnet(len_img, refer, output)
                    # print(out.shape)
                    out = out.permute(0, 2, 1)
                    
                    out = out.reshape([-1, 136, 1])
                    # out = out.squeeze(-1)
                    all_result = out
                    all_result = all_result.reshape([-1, 68, 2])
                    all_result = all_result * 160
                    result_list = []
                    for p in range(all_result.size(0)):
                        pred = all_result[p,:,:]
                        pred = pred.cpu().numpy()
                        dpi = 100
                        fig = plt.figure(figsize=(160/dpi, 160/dpi), dpi = dpi)
                        ax = fig.add_subplot(1,1,1)
                        ax.imshow(np.ones((160,160,3)))
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                        #chin
                        ax.plot(pred[0:17,0],pred[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
                        #left and right eyebrow
                        ax.plot(pred[17:22,0],pred[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
                        ax.plot(pred[22:27,0],pred[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
                        #nose
                        ax.plot(pred[27:31,0],pred[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
                        ax.plot(pred[31:36,0],pred[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
                        #left and right eye
                        ax.plot(pred[36:42,0],pred[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
                        ax.plot(pred[42:48,0],pred[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
                        #outer and iner lip
                        ax.plot(pred[48:60,0],pred[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
                        ax.plot(pred[60:68,0],pred[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
                        ax.axis('off')

                        fig.canvas.draw()
                        #plt.savefig('result/result_{}.png'.format(p), dpi=dpi)

                        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        
                        plt.close(fig)
                        save_img= Image.fromarray(data)
                        save_img.save('result6/result/result_{}_{}.png'.format(i, p), 'PNG')
                    # sample = vqvae.decoder()

                    preds_list = []
                    for pp in range(out.size(0)):

                        all_preds = all_preds.reshape([-1, 68, 2, 1])
                        all_preds = all_preds.squeeze(-1)
                        pred1 = all_preds[pp,:,:] * 160
                        pred1 = pred1.cpu().numpy()
                        
                        dpi = 100
                        fig = plt.figure(figsize=(160/dpi, 160/dpi), dpi = dpi)
                        ax = fig.add_subplot(1,1,1)
                        ax.imshow(np.ones((160,160,3)))
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                        #chin
                        ax.plot(pred1[0:17,0],pred1[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
                        #left and right eyebrow
                        ax.plot(pred1[17:22,0],pred1[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
                        ax.plot(pred1[22:27,0],pred1[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
                        #nose
                        ax.plot(pred1[27:31,0],pred1[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
                        ax.plot(pred1[31:36,0],pred1[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
                        #left and right eye
                        ax.plot(pred1[36:42,0],pred1[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
                        ax.plot(pred1[42:48,0],pred1[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
                        #outer and iner lip
                        ax.plot(pred1[48:60,0],pred1[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
                        ax.plot(pred1[60:68,0],pred1[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
                        ax.axis('off')

                        fig.canvas.draw()
                        #plt.savefig('result/result_{}.png'.format(p), dpi=dpi)

                        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        
                        plt.close(fig)
                        save_result= Image.fromarray(data)
                        save_result.save('result6/preds/preds_{}_{}.png'.format(i, pp ), 'PNG')
                        print('preds_{}_{}.png saved.'.format(i, pp))


def main(args):
    device = torch.device("cuda:0")
    
    args.distributed = dist.get_world_size() > 1

    dataset = LRS2_Dataset(args.path)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)

    loader = DataLoader(
        dataset, batch_size=1 // args.n_gpu, sampler=sampler, num_workers=2
    )
    PATH = "checkpoint/audio2landmark_masked_091.pt"
    checkpoint = torch.load(PATH)
    model = DSNet().to(device)
    optimizerB = optim.Adam(model.parameters(), lr=args.lr)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizerB.load_state_dict(checkpoint["optimizerB_state_dict"])

    model.eval()

    wav2vec2_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    wav2vec2_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device)
    # autoencoder = Autoencoder().to(device)
    criterion = nn.MSELoss()
    # optimizer_W = torch.optim.Adam(dsnet.parameters(), lr=0.0001)


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizerB,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    test(1, loader, wav2vec2_processor, wav2vec2_model, model, optimizerB, scheduler, device)
    


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
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--path", type=str, default='./lrs2_preprocessed2/')

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))



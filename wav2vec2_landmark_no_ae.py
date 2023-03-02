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
        
        self.linear_seq = nn.Conv1d(136, 512, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        # audio_output 3 channel = [1, 3, 512] -> [1, 1, 512]
    
        self.linear1 = nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv1d(128, 136, kernel_size=3, stride=1, padding=1)


    def forward(self, len_img, refer, x):
        try:
            ref = self.linear_seq(refer)    # ref: [1, 512, 1] x: [1, len, 512]
        except:
            print(refer)
        else:
            x = x.permute(0, 2, 1)
            in_seq = torch.cat([ref, x], dim=2)
            in_seq = in_seq.permute(0, 2, 1)
            len_audio = in_seq.size(1)
            # all_seq = torch.zeros((len_audio, 64, 1)).to(device)
            splits = in_seq.split(2, dim=1)
            all_seq = torch.zeros((len(splits), 136, 1)).to(device)
            for c, s in enumerate(splits):
                if s.size(1) == 2:
                    s = self.linear1(s)
                    s = self.ReLU(s)
                    s_t = s.permute(0, 2, 1)
                    s_t = self.linear2(s_t)
                    s_t = self.ReLU(s_t)
                    s_t = self.linear3(s_t)
                    s_t = self.ReLU(s_t)
                    s_t = self.linear4(s_t)
                    
                    all_seq[c] = s_t # all_seq: [66, 64, 1] s_t: [1, 64, 1]
        
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


def train(epoch, loader, wav2vec2_processor, wav2vec2_model, dsnet, optimizerB, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()
    
    dsnet.train()
    dsnet.zero_grad()

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
        refer = []
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
                if t == 0:
                    refer = y
                t2 = time.time()
            if isinstance(refer, list) == True:
                break

        optimizerB.zero_grad()

        if isinstance(refer, list) == True:
            break

        ds_result = dsnet(len_img, refer, output)
        # ds_result = ds_result[:len_img,:,:]
        
        cnt2 += 1

        #print("ds_result: ", ds_result.shape)
        #optimizer_W = torch.optim.Adam(dsnet.parameters(), lr=0.0001)
        loss2 = criterion(ds_result, all_preds)
        wav2vec_loss += loss2.item()
        wav2vec_loss_now = wav2vec_loss / cnt2
        loss2.backward()
        optimizerB.step()

        if scheduler is not None:
            scheduler.step()

        if dist.is_primary():
            lr = optimizerB.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; loss: {wav2vec_loss_now:.4f};"
                    f"latent: {loss2.item():.3f}; "
                    f"lr: {lr:.5f}"
                )
            )
        
            if i % 100 == 0:
                dsnet.eval()

                # waveform = audio.load_wav(audio_path[0][0], sr=16000)
                waveform = audio.load_wav(audio_path[0], sr=16000) 
                input_value = wav2vec2_processor(waveform, return_tensors='pt').input_values
                input_value = input_value.to(device)
                audio_output = wav2vec2_model(input_value).extract_features
                
                # audio_output = audio_output.transpose(1, 2)

                len_img = len(img_list)
                all_preds = torch.zeros((len_img, 136, 1))
                refer = []
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
                        preds_tensor = torch.tensor(preds).to(device)
                        y = preds_tensor.reshape(1, 136)
                        y = y.unsqueeze(-1).to(device)
                        all_preds[t] = y
                        if t == 0:
                            refer = y
                        
                        t2 = time.time()

                with torch.no_grad():

                    out = dsnet(refer, audio_output)
                    # print(out.shape)
                    out = out.permute(0, 2, 1)
                    
                    out = out.reshape([-1, 68, 2, 1])
                    out = out.squeeze(-1)
                    out = out * 160
                    result_list = []
                    for p in range(out.size(0)):
                        pred = out[p,:,:]
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
                        if p < 10:
                            result_list.append(data)
                        plt.close(fig)
                    save_img= Image.fromarray(result_list)
                    save_img.save('result3/result_{}.png'.format(i), 'PNG')
                    # sample = vqvae.decoder()

                    for p in range(out.size(0)):
                        pred = preds[p,:,:]
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
                        if p < 10:
                            result_list.append(data)
                        plt.close(fig)
                    save_img= Image.fromarray(result_list)
                    save_img.save('result3/preds_{}.png'.format(i), 'PNG')
                    




def main(args):
    device = torch.device("cuda:0")
    
    args.distributed = dist.get_world_size() > 1

    dataset = LRS2_Dataset(args.path)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=1 // args.n_gpu, sampler=sampler, num_workers=2
    )
    model = DSNet().to(device)
    optimizerB = optim.Adam(model.parameters(), lr=args.lr)
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



    for i in range(args.epoch):
        SAVE_PATH = f"checkpoint/audio2landmark_dsnet_{str(i + 1).zfill(3)}.pt"

        train(i, loader, wav2vec2_processor, wav2vec2_model, model, optimizerB, scheduler, device)

        if dist.is_primary():
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizerB_state_dict": optimizerB.state_dict(),
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
    parser.add_argument("--path", type=str, default='./lrs2_preprocessed2/')

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))



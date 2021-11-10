# Author: Yuqin Cao
# Email: caoyuqin@sjtu.edu.cn
# Date: 2021/11/8
from torchvision import transforms, models
from torch.utils.data import Dataset
import skvideo
skvideo.setFFmpegPath(r'.\ffmpeg-N-99888-g5c7823ff1c-win64-gpl\bin')
import skvideo.io

import skvideo.io
from PIL import Image
from argparse import ArgumentParser
import math
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from scipy.io import wavfile
from scipy import signal

_PI = np.pi
class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_video_features(dis_video_data, position, sal_index, device='cuda'):
    """video feature extraction"""
    extractor = ResNet50().to(device)
    dis_output = torch.Tensor().to(device)
    extractor.eval()

    ipatch = 0
    with torch.no_grad():
        for iframe in range(0, 192, 2):
            sal_row = int(iframe / 2)
            # initialize
            dis_output1 = torch.Tensor().to(device)
            dis_output2 = torch.Tensor().to(device)

            for idx in range(25):
                patch_idx = sal_index[sal_row, idx]
                dis_batch = dis_video_data[iframe:iframe + 1, 0:3,
                            position[0][patch_idx]:position[0][patch_idx] + patchSize,
                            position[1][patch_idx]:position[1][patch_idx] + patchSize].to(device)

                dis_features_mean, dis_features_std = extractor(dis_batch)
                dis_output1 = torch.cat((dis_output1, dis_features_mean), 0)
                dis_output2 = torch.cat((dis_output2, dis_features_std), 0)

                ipatch = ipatch + 1
                #print('\r Extracting Feature: iframe: {} ipatch: {} '.format(iframe, ipatch), end=' ')

            dis_output = torch.cat((dis_output, torch.cat(
                (dis_output1.mean(axis=0, keepdim=True), dis_output2.mean(axis=0, keepdim=True)), 1)), 0)
            ipatch = 0

        dis_output = dis_output.squeeze()

    return dis_output

def gga_freq_abs(x, sample_rate, freq):
  lx = len(x)
  pik_term = 2 * _PI * freq / sample_rate
  cos_pik_term = np.cos(pik_term)
  cos_pik_term2 = 2 * np.cos(pik_term)

  s1 = x[0]
  s0 = x[1] + cos_pik_term2 * s1
  s2 = s1
  s1 = s0
  for ind in range(2, lx - 1):
    s0 = x[ind] + cos_pik_term2 * s1 - s2
    s2 = s1
    s1 = s0

  s0 = x[lx - 1] + cos_pik_term2 * s1 - s2

  y = np.sqrt((s0 - s1*cos_pik_term)**2 + (s1 * np.sin(pik_term))**2)
  return y

def spectrogram(x, window, window_overlap, bfs, fs):
  num_blocks = int((len(x) - window_overlap) / (len(window) - window_overlap))
  S = np.empty((len(bfs), num_blocks), dtype=np.float64)
  T = np.empty((num_blocks),dtype=np.float)

  for i in range(num_blocks):
    block = window * x[i * (len(window)-window_overlap): i * (len(window)-window_overlap) + len(window)]
    S[:, i] = gga_freq_abs(block, fs, bfs)
    T[i] = (i * (len(window)-window_overlap) + len(window)/2)/fs

  return S,T

def calcSpectrogram(audiofile):
    fs, audio = wavfile.read(audiofile)
    _, ref_audio = wavfile.read(audiofile)
    audio = (audio + 0.5)/32767.5
    audio = audio[:,0]

    windowsize = round(fs*0.02) # 20ms
    overlap = 0.75 # 75% overlap: a 20ms window every 5ms

    window_overlap = int(windowsize*overlap)
    window = signal.get_window('hamming',windowsize,fftbins='True')

    dim = 224
    bfs = [i for i in np.arange(30,3820+(3820-30)/(dim-1),(3820-30)/(dim-1))]
    bfs = np.array(bfs,dtype=float)
    bfs = 700*(pow(10,(bfs/2595))-1)
    S, t_sp= spectrogram(audio,window,window_overlap,bfs,fs)

    S = abs(np.array(S));                       # remove complex component

    S[(S==0)] = pow(2,-52);                  # no -infs in power dB

    spec_bf= np.zeros((S.shape[0],S.shape[1]))
    for i in range(len(S)):
        for j in range(len(S[i])):
            spec_bf[i][j] = math.log(S[i][j])
    return spec_bf, t_sp

def get_audio_features(audios_data, audio_tStamp, frameRate, video_length, device='cuda'):
    """audio feature extraction"""
    extractor = ResNet50().to(device)
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()


    patchSize = 224
    frameSkip = 2
    with torch.no_grad():
        for iFrame in range(1, video_length,frameSkip):
            tCenter =np.argmin(abs(audio_tStamp - iFrame / frameRate))
            tStart = tCenter - patchSize / 2 + 1
            tEnd = tCenter + patchSize / 2
            if tStart < 1:
                tStart = 1
                tEnd = patchSize
            else:
                if tEnd > audios_data.shape[2]:
                    tStart = audios_data.shape[2] - patchSize + 1
                    tEnd = audios_data.shape[2]
            specRef_patch = audios_data[:, :, int(tStart-1): int(tEnd)]
            refRGB = torch.cat((specRef_patch, specRef_patch, specRef_patch),0)

            last_batch = refRGB.view(1,3,specRef_patch.shape[1],specRef_patch.shape[2]).float().to(device)
            features_mean, features_std = extractor(last_batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)

        output = torch.cat((output1, output2), 1).squeeze()

    return output

class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)
        self.dropout = nn.AlphaDropout(p=dropout_p)
        self.fc = nn.Linear(reduced_size, reduced_size)

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers - 1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input

class ANNAVQA(nn.Module):
    def __init__(self, input_size=4096, min_len=48, reduced_size=2048, hidden_size=1024):
        super(ANNAVQA, self).__init__()
        self.hidden_size = hidden_size
        self.min_len = min_len
        self.video_ann = ANN(input_size, reduced_size, 1)
        self.video_rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.video_q1 = nn.Linear(hidden_size, 512)
        self.video_relu = nn.ReLU()
        self.video_dro = nn.Dropout()
        self.video_q2 = nn.Linear(512, 1)

        self.audio_ann = ANN(input_size, reduced_size, 1)
        self.audio_rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.audio_q1 = nn.Linear(hidden_size, 512)
        self.audio_relu = nn.ReLU()
        self.audio_dro = nn.Dropout()
        self.audio_q2 = nn.Linear(512, 1)

        self.fc1 = nn.Linear(min_len, 32)  #
        self.relu1 = nn.ReLU()
        self.dro1 = nn.Dropout()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dro2 = nn.Dropout()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, input, input_length, video_length):
        video_input = self.video_ann(input[:, :video_length, :])
        audio_input = self.audio_ann(input[:, video_length:input.size(1), :])
        video_outputs, _ = self.video_rnn(video_input, self._get_initial_state(input.size(0), input.device))
        audio_outputs, _ = self.audio_rnn(audio_input, self._get_initial_state(input.size(0), input.device))
        video_q1 = self.video_q1(video_outputs)
        audio_q1 = self.audio_q1(audio_outputs)
        video_relu = self.video_relu(video_q1)
        audio_relu = self.audio_relu(audio_q1)
        video_dro = self.video_dro(video_relu)
        audio_dro = self.audio_dro(audio_relu)
        video_q = self.video_q2(video_dro)
        audio_q = self.audio_q2(audio_dro)
        score = torch.zeros_like(input_length, device=video_q.device)
        for i in range(input_length.shape[0]):
            video_qi = video_q[i, :]
            audio_qi = audio_q[i, :]
            fc1 = self.fc1(torch.cat([video_qi.squeeze(), audio_qi.squeeze()]))
            relu1 = self.relu1(fc1)
            dro1 = self.dro1(relu1)
            fc2 = self.fc2(dro1)
            relu2 = self.relu2(fc2)
            dro2 = self.dro2(relu2)
            score[i] = self.fc3(dro2)
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0

if __name__ == "__main__":
    parser = ArgumentParser(description='"Test Demo of ANNAVQA')
    parser.add_argument("--seed", type=int, default=19920524)
    parser.add_argument('--model_path', default='./models/NR_model', type=str,
                        help='model path (default: ./models/NR_model)')
    parser.add_argument('--dis_video_path', default='./dis_test.yuv', type=str,
                        help='video path (default: ./dis_test.yuv)')
    parser.add_argument('--dis_audio_path', default='./dis_test.wav', type=str,
                        help='video path (default: ./dis_test.wav)')
    parser.add_argument('--frame_rate', default=24, type=float,
                        help='Frame Rate')
    parser.add_argument('--video_width', type=int, default=1920,
                        help='video width')
    parser.add_argument('--video_height', type=int, default=1080,
                        help='video height')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    # Video data pre-processing

    dis_video_data = skvideo.io.vread(args.dis_video_path, args.video_height, args.video_width, inputdict={'-pix_fmt': 'yuvj420p'})

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    video_length = 192 #dis_video_data.shape[0]
    video_channel = dis_video_data.shape[3]
    video_height = dis_video_data.shape[1]
    video_width = dis_video_data.shape[2]
    transformed_dis_video = torch.zeros([video_length, video_channel, video_height, video_width])

    for frame_idx in range(192):
        dis_frame = dis_video_data[frame_idx]
        dis_frame = Image.fromarray(dis_frame)
        dis_frame = transform(dis_frame)
        transformed_dis_video[frame_idx] = dis_frame

    dis_patch = math.ceil(video_height/1000)*100

    # Crop image patches
    patchSize = 224
    position_width = []
    position_height = []
    for h in range(0, video_height, dis_patch):
        if h < video_height - patchSize + 1:
            for w in range(0, video_width, dis_patch):
                if w < video_width - patchSize:
                    position_height.append(h)
                    position_width.append(w)
                else:
                    position_height.append(h)
                    position_width.append(video_width - patchSize)
                    break
        else:
            for w in range(0, video_width, dis_patch):
                if w < video_width - patchSize:
                    position_height.append(video_height - patchSize)
                    position_width.append(w)
                else:
                    position_height.append(video_height - patchSize)
                    position_width.append(video_width - patchSize)
                    break
            break

    # Video feature extraction
    position = [position_height, position_width]

    # Using saliency detection results from sal_position.m
    SDdatainfo = './test_position.mat'
    SDInfo = scipy.io.loadmat(SDdatainfo)
    sal_index = SDInfo['sort_frame'] - 1

    dis_video_features = get_video_features(transformed_dis_video, position, sal_index, device)


    #Audio data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    [dis_S, dis_T] = calcSpectrogram(args.dis_audio_path)
    transforms_dis_audio = transform(dis_S)

    dis_audio_features = get_audio_features(transforms_dis_audio, dis_T, args.frame_rate, video_length, device)

    # Quality prediction using ANNAVQA
    seg_num = 4
    tmp_video_length = 96
    min_audio_len = 96
    feat_dim = 4096
    seg_video_len = int(tmp_video_length / seg_num)
    seg_audio_len = int(min_audio_len / seg_num)
    length = np.zeros((1, 1))
    length[0] = seg_video_len + seg_audio_len
    length = torch.from_numpy(length).float()

    model = ANNAVQA()
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    y_pred = 0
    with torch.no_grad():
        for seg_index in range(seg_num):
            features = torch.zeros(1, seg_video_len + seg_audio_len, feat_dim, device=device)
            video_features = dis_video_features[seg_video_len * seg_index:seg_video_len * (seg_index + 1), :feat_dim]
            audio_features = dis_audio_features[seg_audio_len * seg_index:seg_audio_len * (seg_index + 1), :feat_dim]
            Feature = torch.cat([video_features.float(), audio_features.float()], axis=0)
            features[0, :Feature.shape[0], :] = Feature
            y_pred = y_pred + model(features, length, seg_video_len).to('cpu').numpy()

        y_pred = y_pred/seg_num
        print("Predicted quality: {}".format(y_pred[0][0]))

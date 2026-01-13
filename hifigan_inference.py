import utils
import numpy as np
from tqdm import tqdm
import torch
import soundfile as sf
import glob2
import os
import librosa
device = "cuda:0"

def main(mel_path):
    print("Loading vocoder...")
    vocoder = utils.get_vocoder(0)
    vocoder.eval()
    print("Loaded vocoder.")
    mels = get_mels(mel_path)
    for mel in tqdm(mels,total=len(mels)):
        mel_numpy = np.load(mel)
        x = torch.FloatTensor(mel_numpy).to(device)
        if x.size(-1) == 80 :
            x = x.transpose(-1,-2)
        if len(x.size())<3:
            x = torch.unsqueeze(x,0)
        wav_rs = vocoder(x)[0][0].detach().cpu().numpy()
        wav_rs = librosa.resample(wav_rs, orig_sr=22050, target_sr=16000)
        wav_path = mel.replace(".npy",".wav")
        sf.write(wav_path,wav_rs ,16000,subtype='PCM_16')

def get_mels(mel_path):
    mel_path = os.path.join(mel_path,"")
    mels = glob2.glob(mel_path+"**"+"/"+"*.npy")
    tqdm.write(f"{len(mels)}")
    return mels

if __name__ == "__main__":
    main("../syn_mel")
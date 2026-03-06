import math
import multiprocessing
import os
import argparse
from random import shuffle
import torchaudio
import torchaudio.transforms as T

import torch
from glob import glob
from tqdm import tqdm

from audiolm_pytorch import SoundStream, EncodecWrapper
import utils
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa
import numpy as np
from speechtokenizer import SpeechTokenizer
from masked_prosody_model import MaskedProsodyModel
from speechbrain.inference.speaker import EncoderClassifier
from speaker_encoder.voice_encoder import SpeakerEncoder

hps = utils.get_hparams_from_file("config_orig.json")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
in_dir = ""
st_config_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/config.json"    
st_ckpt_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/ckpt.dev"   


def process_one(filename,out_dir, mpm_model, codec,spk_model,freevc_model,speechtokenizer,mpm,spk_emb,freevc_emb):
    wav, sr = torchaudio.load(filename)
    if wav.shape[0] > 1:  # mix to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav16k = T.Resample(sr, 16000)(wav)
    wav24k = T.Resample(sr, 24000)(wav)
    #filename = os.path.join(out_dir,os.path.basename(filename))
    filename = filename.replace(in_dir, out_dir)
    wav24k_path = filename
    # if not os.path.exists(os.path.dirname(wav24k_path)):
        # os.makedirs(os.path.dirname(wav24k_path))
    # torchaudio.save(wav24k_path, wav16k, 16000)
    st_embed_path = filename + ".st_first.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav16k = wav16k.to(device)
    # c = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k[0])
    if speechtokenizer:
        st_first_embed = codec.encode_first(wav16k.unsqueeze(0))
        st_all_codes = codec.encode(wav16k.unsqueeze(0))
        _, st_all_embed = codec.decode(st_all_codes)
    
        torch.save(st_first_embed.cpu(), st_embed_path)
        torch.save(st_all_embed.cpu(), st_embed_path.replace("st_first","st_all"))

    #f0_path = filename + ".f0.npy"
    #f0 = utils.compute_f0_dio(
        #wav24k.cpu().numpy()[0], sampling_rate=24000, hop_length=hop_length
    #)
    #np.save(f0_path, f0)
    if mpm:
        mpm_path = filename.replace(".wav",".mpm.pt")
        tmp_feature =  mpm_model.process_audio(filename, layer=7,device=device)
        torch.save(tmp_feature.cpu(),mpm_path)

    if spk_emb:
        spk_emb_path = filename.replace(".wav",".ecapa.pt")
        embeddings = spk_model.encode_batch(wav16k)
        torch.save(embeddings.cpu(),spk_emb_path)

    if freevc_emb:
        freevc_emb_path = filename.replace(".wav",".freevcs.pt")
        wav_tgt, _ = librosa.load(filename, sr=16000) 
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        g_tgt = freevc_model.embed_utterance(wav_tgt)
        g_tgt = torch.from_numpy(g_tgt).cpu()
        torch.save(g_tgt,freevc_emb_path)
    # spec_path = filename.replace(".wav", ".spec.pt")
    # spec_process = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=24000,
    #     n_fft=1024,
    #     hop_length=256,
    #     n_mels=100,
    #     center=True,
    #     power=1,
    # )
    # spec = spec_process(wav24k)# 1 100 T
    # spec = torch.log(torch.clip(spec, min=1e-7))
    # torch.save(spec, spec_path)


def process_batch(filenames,out_dir,speechtokenizer=True,mpm=True,spk_emb=True,freevc_emb=True):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmodel = utils.get_hubert_model().to(device)
    codec = EncodecWrapper()
    print("Loaded hubert.")
    codec = SpeechTokenizer.load_from_checkpoint(st_config_path,st_ckpt_path)
    codec.eval()
    codec = codec.to(device)
    mpm_model =  MaskedProsodyModel.from_pretrained("cdminix/masked_prosody_model")
    mpm_model = mpm_model.to(device)
    ecapa_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    ecapa_model = ecapa_model.to(device)  

    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')
    smodel = smodel.to(device)

    for filename in tqdm(filenames):
        process_one(filename,out_dir, mpm_model, codec,ecapa_model,smodel,speechtokenizer,mpm,spk_emb,freevc_emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset", help="path to input dir"
    )
    parser.add_argument("--out_dir",type=str,default="exp/data")
    parser.add_argument(
        "--actions",
        nargs="*",
        default=[],
        help="List of actions to enable: speechtokenizer, mpm, spk_emb",
    )
    parser.add_argument("--speechtokenizer", action="store_true")
    parser.add_argument("--mpm", action="store_true")
    parser.add_argument("--spk_emb", action="store_true")
    parser.add_argument("--freevc_emb",action="store_true")
    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/**/**/*.wav", recursive=True)  # [:10]
    in_dir = args.in_dir
    shuffle(filenames)
    actions = {name.lower() for name in args.actions}
    speechtokenizer = "speechtokenizer" in actions if actions else args.speechtokenizer
    mpm = "mpm" in actions if actions else args.mpm
    spk_emb = "spk_emb" in actions if actions else args.spk_emb
    freevc_emb = "freevc_emb" in actions if actions else args.freevc_emb
    process_batch(filenames, args.out_dir, speechtokenizer, mpm, spk_emb,freevc_emb)

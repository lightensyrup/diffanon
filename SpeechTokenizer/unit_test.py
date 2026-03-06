import os 
from speechtokenizer import SpeechTokenizer
import torch
import torchaudio

if __name__ == "__main__":
    config_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/config.json"    
    ckpt_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/ckpt.dev"   

    wav_example = "diffanon_repo/exp/data/libritts_100/LibriTTS/train-clean-100/7113/86041/7113_86041_000003_000008.wav"

    model = SpeechTokenizer.load_from_checkpoint(config_path,ckpt_path)
    model.eval()

    # Load and pre-process speech waveform
    wav, sr = torchaudio.load(wav_example)

    # monophonic checking
    if wav.size(0) > 1:
        wav = wav[:1,:]

    if sr != model.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)

    wav = wav.unsqueeze(0)

    # Extract discrete codes from SpeechTokenizer
    with torch.no_grad():
        codes = model.encode(wav) 
        embeds = model.encode_first(wav)
        wav_onlyfirst = model.decode_first(embeds).squeeze(0)
    
    torchaudio.save("deneme2.wav",wav_onlyfirst,sample_rate=16000)
    deneme = "a"
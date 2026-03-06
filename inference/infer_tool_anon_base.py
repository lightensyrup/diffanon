import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from inference import slicer
import gc
from ema_pytorch import EMA

import librosa
import numpy as np
# import onnxruntime
import soundfile
import torch
import torchaudio
from vocos import Vocos
import torchaudio.transforms as T
from torch.nn.functional import cosine_similarity

from accelerate import Accelerator
import utils
from model_anon import DiffAnon, Trainer

from speechtokenizer import SpeechTokenizer
from masked_prosody_model import MaskedProsodyModel
from speechbrain.inference.speaker import EncoderClassifier

st_config_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/config.json"    
st_ckpt_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/ckpt.dev"   

SIM_THRESHOLD = 0.5

logging.getLogger('matplotlib').setLevel(logging.WARNING)
def load_mod(model_path, device, cfg):
    data = torch.load(model_path, map_location=device)
    model = DiffAnon(cfg=cfg)
    model.load_state_dict(data['model'])

    ema = EMA(model)
    ema.to(device)
    ema.load_state_dict(data["ema"])
    return ema.ema_model


def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.replace("\\", "/").split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()

def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])

def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), 'constant', constant_values=(0, 0))
        return padded_arr
    
def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i-pre if i-pre>=0 else i: i + n]


class F0FilterException(Exception):
    pass

class Svc(object):
    def __init__(self, model_path, config_path,
                 device=None,
                 ):
        self.model_path = model_path
        def _cuda_available():
            try:
                return torch.cuda.is_available()
            except RuntimeError as err:
                if "cudaGetDeviceCount" in str(err):
                    logging.warning("CUDA availability check failed (%s); defaulting to CPU.", err)
                    return False
                raise

        if device is None:
            self.dev = torch.device("cuda" if _cuda_available() else "cpu")
        else:
            requested_device = torch.device(device)
            if requested_device.type == "cuda" and not _cuda_available():
                logging.warning("CUDA device requested but unavailable; falling back to CPU.")
                self.dev = torch.device("cpu")
            else:
                self.dev = requested_device
        self.model = None
        self.cfg = json.load(open(config_path))
        self.target_sample = self.cfg['data']['sampling_rate']
        self.hop_size = self.cfg['data']['hop_length']
        # load hubert
        # self.hubert_model = utils.get_hubert_model().to(self.dev)
        self.codec = SpeechTokenizer.load_from_checkpoint(st_config_path,st_ckpt_path)
        self.codec.eval()
        self.codec = self.codec.to(self.dev)
        self.mpm_model =  MaskedProsodyModel.from_pretrained("cdminix/masked_prosody_model")
        self.mpm_model = self.mpm_model.to(self.dev)
        self.ecapa_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        self.ecapa_model = self.ecapa_model.to(self.dev)  
    

        self.load_model()
        # self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        # self.vocos = Vocos.from_pretrained("/home/anon/.cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21")

    def load_model(self):
        self.model = load_mod(self.model_path, self.dev, self.cfg)
        self.model.eval()

    def get_input_features(self, in_path,  refer_path,cr_threshold=0.05):
        # c, refer, f0, uv, lengths, refer_lengths
        wav, sr = librosa.load(in_path, sr=self.target_sample)
        selected = False
        # if F0_mean_pooling == True:
        #     f0, uv = utils.compute_f0_uv_torchcrepe(torch.FloatTensor(wav), sampling_rate=self.target_sample, hop_length=self.hop_size,device=self.dev,cr_threshold = cr_threshold)
        #     if f0_filter and sum(f0) == 0:
        #         raise F0FilterException("No voice detected")
        #     f0 = torch.FloatTensor(list(f0))
        #     uv = torch.FloatTensor(list(uv))
        # if F0_mean_pooling == False:
        #     f0 = utils.compute_f0_parselmouth(wav, sampling_rate=self.target_sample, hop_length=self.hop_size)
        #     if f0_filter and sum(f0) == 0:
        #         raise F0FilterException("No voice detected")
        #     f0, uv = utils.interpolate_f0(f0)
        #     f0 = torch.FloatTensor(f0)
        #     uv = torch.FloatTensor(uv)

        # f0 = f0 * 2 ** (tran / 12)
        # f0 = f0.unsqueeze(0).to(self.dev)
        # uv = uv.unsqueeze(0).to(self.dev)


        wav16k = librosa.resample(wav, orig_sr=self.target_sample, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(self.dev)
        # c = utils.get_hubert_content(self.hubert_model, wav_16k_tensor=wav16k)
        # c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1])

        c = self.codec.encode_first(wav16k.unsqueeze(0).unsqueeze(0))
        c = c.to(self.dev)

        f0 =  self.mpm_model.process_audio(in_path, layer=7,device=self.dev)
        f0 = torch.permute(f0,(1,0))
        f0 = torch.nn.functional.interpolate(f0.unsqueeze(0),size = c.shape[2],mode="linear")
        f0 = f0.to(self.dev)

        spec_process = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=100,
            center=True,
            power=1,
        )
         

        # TODO (anon) check dimensions
                # TODO (extract ecapa-tdnn)
        #TODO (anon) check dimensions
        if refer_path is not None:
            refer_wav, sr = torchaudio.load(refer_path)
            wav24k = T.Resample(sr, 24000)(refer_wav)
            spec = spec_process(wav24k)# 1 100 T
            spec = torch.log(torch.clip(spec, min=1e-7))
            refer = spec.to(self.dev)


            ref_wav, ref_sr = librosa.load(refer_path, sr=self.target_sample)
            ref_wav16k = librosa.resample(ref_wav, orig_sr=self.target_sample, target_sr=16000)
            ref_wav16k = torch.from_numpy(ref_wav16k).to(self.dev)
            #
            spk_emb = self.ecapa_model.encode_batch(ref_wav16k)
            spk_emb = torch.nn.functional.normalize(spk_emb)
            if torch.isnan(spk_emb).any():
                print("Spk emb contains Nan!!!")
            #TODO (anon) normalize speaker embedding
            spk_emb = spk_emb.repeat(c.shape[2],1,1).permute(1,2,0) 
            spk_emb = spk_emb.to(self.dev)
            #
        else:
            refer_wav, sr = torchaudio.load(in_path)
            wav24k = T.Resample(sr, 24000)(refer_wav)
            spec = spec_process(wav24k)# 1 100 T
            spec = torch.log(torch.clip(spec, min=1e-7))
            refer = spec.to(self.dev)

            # ref_wav16k = torch.from_numpy(ref_wav16k).to(self.dev)

            spk_emb_orig = self.ecapa_model.encode_batch(wav16k)
            spk_emb_orig = torch.nn.functional.normalize(spk_emb_orig)
            #

            while not selected:

                dim = 192  # or 256 depending on your ECAPA model
                spk_emb = torch.randn(dim)
                spk_emb = spk_emb.unsqueeze(0).unsqueeze(0)
                spk_emb = spk_emb / spk_emb.norm(p=2)  # unit norm
                tmp_sim = cosine_similarity(spk_emb_orig.squeeze(0),spk_emb.squeeze(0))
                if tmp_sim.item() < SIM_THRESHOLD:
                    selected = True

            spk_emb = spk_emb.repeat(c.shape[2],1,1).permute(1,2,0)  
            spk_emb = spk_emb.to(self.dev)
            # print(x.shape, x.norm().item())
        lengths = torch.LongTensor([c.shape[2]]).to(self.dev)
        refer_lengths = torch.LongTensor([refer.shape[2]]).to(self.dev)

        return c, spk_emb, f0,refer,  lengths, refer_lengths




    def get_unit_f0_code(self, in_path, tran, refer_path, f0_filter ,F0_mean_pooling,cr_threshold=0.05):
        # c, refer, f0, uv, lengths, refer_lengths
        wav, sr = librosa.load(in_path, sr=self.target_sample)

        if F0_mean_pooling == True:
            f0, uv = utils.compute_f0_uv_torchcrepe(torch.FloatTensor(wav), sampling_rate=self.target_sample, hop_length=self.hop_size,device=self.dev,cr_threshold = cr_threshold)
            if f0_filter and sum(f0) == 0:
                raise F0FilterException("No voice detected")
            f0 = torch.FloatTensor(list(f0))
            uv = torch.FloatTensor(list(uv))
        if F0_mean_pooling == False:
            f0 = utils.compute_f0_parselmouth(wav, sampling_rate=self.target_sample, hop_length=self.hop_size)
            if f0_filter and sum(f0) == 0:
                raise F0FilterException("No voice detected")
            f0, uv = utils.interpolate_f0(f0)
            f0 = torch.FloatTensor(f0)
            uv = torch.FloatTensor(uv)

        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0).to(self.dev)
        uv = uv.unsqueeze(0).to(self.dev)

        #TODO (anon) extract prosody features

        wav16k = librosa.resample(wav, orig_sr=self.target_sample, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(self.dev)
        c = utils.get_hubert_content(self.hubert_model, wav_16k_tensor=wav16k)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1])

        #TODO (anon) extract first-level speechtokenizer
        c = c.unsqueeze(0).to(self.dev)

        refer_wav, sr = torchaudio.load(refer_path)
        wav24k = T.Resample(sr, 24000)(refer_wav)
        spec_process = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=100,
            center=True,
            power=1,
        )
        spec = spec_process(wav24k)# 1 100 T
        spec = torch.log(torch.clip(spec, min=1e-7))
        refer = spec.to(self.dev)

        lengths = torch.LongTensor([c.shape[2]]).to(self.dev)
        refer_lengths = torch.LongTensor([refer.shape[2]]).to(self.dev)

        return c, refer, f0, uv, lengths, refer_lengths

    def infer(self, tran,
            raw_path,
            refer_path,
            auto_predict_f0=False,
            f0_filter=False,
            F0_mean_pooling=False,
            cr_threshold = 0.05
        ):

        c, refer, f0, uv, lengths, refer_lengths = self.get_unit_f0_code(raw_path, tran, refer_path, f0_filter,F0_mean_pooling,cr_threshold=cr_threshold)
        #TODO(anon) the feature extraction for input is get_unit_f0_code, modify it for our features
        with torch.no_grad():
            start = time.time()
            audio = self.model.sample(c, refer, f0, uv, lengths, refer_lengths, self.vocos, auto_predict_f0 =auto_predict_f0,sampling_timesteps=100)[0].detach().cpu()
            #TODO(anon) add speaker embedding here instead of refer
            # print(audio.shape)
            use_time = time.time() - start
            print("ns2vc use time:{}".format(use_time))
        return audio, audio.shape[-1]

    #TODO (anon) implement standard and cfg based implementation

    def infer_anon(self,
            raw_path,
            refer_path,
            auto_predict_f0=False,
            cr_threshold = 0.05,
            use_cond = [1,1],
            sampling='ddim',
            cfg_scales = None
        ):

        # c, refer, f0, uv, lengths, refer_lengths = self.get_unit_f0_code(raw_path, tran, refer_path, f0_filter,F0_mean_pooling,cr_threshold=cr_threshold)

        c, spk_emb, f0,refer, lengths, refer_lengths = self.get_input_features(raw_path, refer_path, cr_threshold)
        #TODO(anon) the feature extraction for input is get_unit_f0_code, modify it for our features
        with torch.no_grad():
            start = time.time()
            audio = self.model.sample(c, spk_emb,refer, f0, lengths, refer_lengths, self.codec,sample_method=sampling, auto_predict_f0 =auto_predict_f0,sampling_timesteps=100,use_cond=use_cond,cfg_scales=cfg_scales)[0].detach().cpu()
            #TODO(anon) add speaker embedding here instead of refer
            # print(audio.shape)
            use_time = time.time() - start
            print("ns2vc use time:{}".format(use_time))
        return audio, audio.shape[-1]

    def clear_empty(self):
        # clean up vram
        torch.cuda.empty_cache()

    def unload_model(self):
        # unload model
        self.model = self.model.to("cpu")
        del self.model
        gc.collect()

    def slice_inference(self,
                        raw_audio_path,
                        spk,
                        tran,
                        slice_db,
                        cluster_infer_ratio,
                        auto_predict_f0,
                        noice_scale,
                        pad_seconds=0.5,
                        clip_seconds=0,
                        lg_num=0,
                        lgr_num =0.75,
                        F0_mean_pooling = False,
                        enhancer_adaptive_key = 0,
                        cr_threshold = 0.05
                        ):
        wav_path = raw_audio_path
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        per_size = int(clip_seconds*audio_sr)
        lg_size = int(lg_num*audio_sr)
        lg_size_r = int(lg_size*lgr_num)
        lg_size_c_l = (lg_size-lg_size_r)//2
        lg_size_c_r = lg_size-lg_size_r-lg_size_c_l
        lg = np.linspace(0,1,lg_size_r) if lg_size!=0 else 0
        
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            # padd
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
                audio.extend(list(pad_array(_audio, length)))
                continue
            if per_size != 0:
                datas = split_list_by_n(data, per_size,lg_size)
            else:
                datas = [data]
            for k,dat in enumerate(datas):
                per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample)) if clip_seconds!=0 else length
                if clip_seconds!=0: print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                # padd
                pad_len = int(audio_sr * pad_seconds)
                dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)
                out_audio, out_sr = self.infer(spk, tran, raw_path,
                                                    cluster_infer_ratio=cluster_infer_ratio,
                                                    auto_predict_f0=auto_predict_f0,
                                                    noice_scale=noice_scale,
                                                    F0_mean_pooling = F0_mean_pooling,
                                                    enhancer_adaptive_key = enhancer_adaptive_key,
                                                    cr_threshold = cr_threshold
                                                    )
                _audio = out_audio.cpu().numpy()
                pad_len = int(self.target_sample * pad_seconds)
                _audio = _audio[pad_len:-pad_len]
                _audio = pad_array(_audio, per_length)
                if lg_size!=0 and k!=0:
                    lg1 = audio[-(lg_size_r+lg_size_c_r):-lg_size_c_r] if lgr_num != 1 else audio[-lg_size:]
                    lg2 = _audio[lg_size_c_l:lg_size_c_l+lg_size_r]  if lgr_num != 1 else _audio[0:lg_size]
                    lg_pre = lg1*(1-lg)+lg2*lg
                    audio = audio[0:-(lg_size_r+lg_size_c_r)] if lgr_num != 1 else audio[0:-lg_size]
                    audio.extend(lg_pre)
                    _audio = _audio[lg_size_c_l+lg_size_r:] if lgr_num != 1 else _audio[lg_size:]
                audio.extend(list(_audio))
        return np.array(audio)

class RealTimeVC:
    def __init__(self):
        self.last_chunk = None
        self.last_o = None
        self.chunk_len = 16000  # chunk length
        self.pre_len = 3840  # cross fade length, multiples of 640

    # Input and output are 1-dimensional numpy waveform arrays

    def process(self, svc_model, speaker_id, f_pitch_change, input_wav_path,
                cluster_infer_ratio=0,
                auto_predict_f0=False,
                noice_scale=0.4,
                f0_filter=False):

        import maad
        audio, sr = torchaudio.load(input_wav_path)
        audio = audio.cpu().numpy()[0]
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            input_wav_path.seek(0)

            audio, sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path,
                                        cluster_infer_ratio=cluster_infer_ratio,
                                        auto_predict_f0=auto_predict_f0,
                                        noice_scale=noice_scale,
                                        f0_filter=f0_filter)

            audio = audio.cpu().numpy()
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return audio[-self.chunk_len:]
        else:
            audio = np.concatenate([self.last_chunk, audio])
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)

            audio, sr = svc_model.infer(speaker_id, f_pitch_change, temp_wav,
                                        cluster_infer_ratio=cluster_infer_ratio,
                                        auto_predict_f0=auto_predict_f0,
                                        noice_scale=noice_scale,
                                        f0_filter=f0_filter)

            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return ret[self.chunk_len:2 * self.chunk_len]

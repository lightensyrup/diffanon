import os
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from masked_prosody_model import MaskedProsodyModel, ModelArgs
from masked_prosody_model.modules import PositionalEncoding, TransformerEncoder, ConformerLayer
from masked_prosody_model.measures import PitchMeasure, EnergyMeasure, VoiceActivityMeasure

@pytest.fixture
def model_args():
    return ModelArgs(
        n_layers=8,
        n_heads=2,
        kernel_size=7,
        filter_size=256,
        hidden_dim=512,
        dropout=0.1,
        pitch_min=50,
        pitch_max=300,
        energy_min=0,
        energy_max=0.2,
        vad_min=0,
        vad_max=1,
        bins=128
    )

@pytest.fixture
def model(model_args):
    return MaskedProsodyModel.from_pretrained("cdminix/masked_prosody_model")

@pytest.fixture
def dummy_input():
    return torch.randint(0, 1, (1, 3, 517))  # 3 features: pitch, energy, vad

def test_model_initialization(model_args):
    model = MaskedProsodyModel(model_args)
    assert isinstance(model, MaskedProsodyModel)
    assert model.args == model_args
    assert isinstance(model.pitch_embedding, torch.nn.Embedding)
    assert isinstance(model.energy_embedding, torch.nn.Embedding)
    assert isinstance(model.vad_embedding, torch.nn.Embedding)
    assert isinstance(model.transformer, TransformerEncoder)

def test_forward_pass(model, dummy_input):
    outputs = model(dummy_input)
    assert isinstance(outputs, list)
    assert len(outputs) == 3  # pitch, energy, vad outputs
    assert all(isinstance(x, torch.Tensor) for x in outputs)
    assert outputs[0].shape == torch.Size([1, 517, 128])  # batch_size, seq_len, bins
    assert outputs[1].shape == torch.Size([1, 517, 128])  # batch_size, seq_len, bins
    assert outputs[2].shape == torch.Size([1, 517, 1])  # batch_size, seq_len, 1

def test_forward_pass_with_layer_return(model, dummy_input):
    outputs = model(dummy_input, return_layer=0)
    assert isinstance(outputs, dict)
    assert "y" in outputs
    assert "representations" in outputs
    assert len(outputs["y"]) == 3
    assert isinstance(outputs["representations"], torch.Tensor)

def test_positional_encoding():
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    x = torch.randn(2, 50, d_model)
    output = pe(x)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_conformer_layer():
    d_model = 64
    n_heads = 2
    layer = ConformerLayer(
        d_model,
        n_heads,
        conv_in=d_model,
        conv_filter_size=d_model,
        conv_kernel=(3, 1),
        batch_first=True
    )
    x = torch.randn(2, 10, d_model)
    output = layer(x)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_measures():
    # Test pitch measure
    pitch_measure = PitchMeasure()
    audio = np.random.randn(22050)  # 1 second of audio
    durations = np.array([22050])
    pitch_result = pitch_measure(audio, durations)
    assert "measure" in pitch_result
    assert isinstance(pitch_result["measure"], np.ndarray)

    # Test energy measure
    energy_measure = EnergyMeasure()
    energy_result = energy_measure(audio, durations)
    assert "measure" in energy_result
    assert isinstance(energy_result["measure"], np.ndarray)

    # Test VAD measure
    vad_measure = VoiceActivityMeasure()
    vad_result = vad_measure(audio, durations)
    assert "measure" in vad_result
    assert isinstance(vad_result["measure"], np.ndarray)
    assert np.all(np.isin(vad_result["measure"], [0, 1]))  # Binary values

def test_model_save_load(model, model_args, dummy_input):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model
        model.save_model(tmpdir)
        
        # Check if files exist
        assert os.path.exists(os.path.join(tmpdir, "pytorch_model.bin"))
        assert os.path.exists(os.path.join(tmpdir, "model_config.yml"))
        
        # Load model
        loaded_model = MaskedProsodyModel.from_pretrained(tmpdir)
        
        # Check if loaded model matches original
        assert isinstance(loaded_model, MaskedProsodyModel)
        
        assert loaded_model.args.bins == model_args.bins

def test_full_pipeline(model):
    audio_path = "tests/test_audio.wav"
    representation = model.process_audio(audio_path, layer=7)
    assert representation.shape == torch.Size([727, 256])
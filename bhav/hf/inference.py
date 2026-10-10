"""BhavVaani Whisper SER: 4-class Hindi speech emotion recognition (angry, happy, neutral, sad).

Usage:
    from inference import BhavWhisperSER
    model = BhavWhisperSER.from_pretrained('<hf repo id or local dir>')
    model.predict(['clip1.wav', 'clip2.wav'])      # -> list of {label: prob}

The frozen bottom of the Whisper-large-v3 encoder runs as in pre-training (30 s padding); the fine-tuned top
layers, final LayerNorm and an attentive-statistics pooling head (head_and_top_layers.pt) run on the valid frames.
"""
import copy, json, os
import numpy as np, torch, torch.nn as nn, librosa
from transformers import WhisperFeatureExtractor, WhisperModel


class _Head(nn.Module):
    def __init__(self, layers, ln, D, drop=0.3):
        super().__init__()
        self.layers, self.ln = layers, ln
        self.att = nn.Sequential(nn.Linear(D, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * D, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 4))

    def forward(self, x, mask):
        am = (1.0 - mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min
        for lyr in self.layers:
            kw = {'layer_head_mask': None} if 'layer_head_mask' in lyr.forward.__code__.co_varnames else {}
            out = lyr(x, attention_mask=am, **kw)
            x = out[0] if isinstance(out, tuple) else out
        h = self.ln(x).float()
        a = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e4).softmax(1)[..., None]
        mu = (a * h).sum(1); sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-5).sqrt()
        return self.head(torch.cat([mu, sd], 1))


class BhavWhisperSER:
    def __init__(self, cfg, state, device=None):
        self.cfg = cfg; self.classes = cfg['classes']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dt = torch.float16 if self.device == 'cuda' else torch.float32
        self.fe = WhisperFeatureExtractor.from_pretrained(cfg['backbone'])
        enc = WhisperModel.from_pretrained(cfg['backbone'], torch_dtype=dt).encoder.eval()
        cut = cfg['frozen_layers']
        head = _Head(copy.deepcopy(nn.ModuleList(enc.layers[cut:])), copy.deepcopy(enc.layer_norm), cfg['d_model'], cfg.get('drop', 0.3))
        head.load_state_dict({k: v.float() for k, v in state.items()})
        enc.layers = enc.layers[:cut]                     # frozen bottom only
        enc.layer_norm = nn.Identity()                    # training cached the un-normalised output of layer `cut`
        self.enc, self.head, self.cut, self.dt = enc.to(self.device), head.float().to(self.device).eval(), cut, dt

    @classmethod
    def from_pretrained(cls, path, device=None):
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path)
        cfg = json.load(open(f'{path}/config.json'))
        state = torch.load(f'{path}/head_and_top_layers.pt', map_location='cpu')
        return cls(cfg, state, device)

    @torch.no_grad()
    def logits(self, wav):
        wav = np.asarray(wav, np.float32)[:30 * 16000]
        f = self.fe(wav, sampling_rate=16000, return_tensors='pt').input_features.to(self.device, self.dt)
        h = self.enc(f, output_hidden_states=True).hidden_states[self.cut]
        T = max(1, int(np.ceil(len(wav) / 320)))
        x = h[:, :T].float(); mask = torch.ones(1, T, device=self.device)
        return self.head(x, mask)[0]

    def predict(self, paths):
        out = []
        for p in paths:
            wav = librosa.load(p, sr=16000)[0] if isinstance(p, str) else p
            pr = self.logits(wav).softmax(0).cpu().numpy()
            out.append({c: float(v) for c, v in zip(self.classes, pr)})
        return out

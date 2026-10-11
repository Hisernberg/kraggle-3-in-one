"""BhavVaani Whisper SER: 4-class Hindi speech emotion recognition (angry, happy, neutral, sad).

    from inference import BhavWhisperSER
    model = BhavWhisperSER.from_pretrained('<hf repo id or local dir>')
    model.predict(['clip1.wav', 'clip2.wav'])      # -> list of {label: prob}

Frozen `openai/whisper-large-v3` encoder (30 s padded log-mel input, as in pre-training); its final-layer output on
the frames that belong to the audio goes through an attentive-statistics pooling head (head.pt, 1.4 MB).
"""
import json, os
import numpy as np, torch, torch.nn as nn, librosa
from transformers import WhisperFeatureExtractor, WhisperModel


class Head(nn.Module):
    def __init__(self, D, H=256, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Dropout(p), nn.Linear(D, H), nn.GELU())
        self.att = nn.Sequential(nn.Linear(H, 64), nn.Tanh(), nn.Linear(64, 1))
        self.out = nn.Sequential(nn.Dropout(p), nn.Linear(2 * H, 4))

    def forward(self, x, mask):                    # x B,T,D  mask B,T (1 = valid frame)
        h = self.inp(x)
        a = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e4).softmax(1)[..., None]
        mu = (a * h).sum(1); sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-5).sqrt()
        return self.out(torch.cat([mu, sd], 1))


class BhavWhisperSER:
    def __init__(self, cfg, state, device=None):
        self.cfg, self.classes = cfg, cfg['classes']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dt = torch.float16 if self.device == 'cuda' else torch.float32
        self.fe = WhisperFeatureExtractor.from_pretrained(cfg['backbone'])
        self.enc = WhisperModel.from_pretrained(cfg['backbone'], torch_dtype=self.dt).encoder.to(self.device).eval()
        self.head = Head(cfg['d_model'], cfg['hidden'], cfg['dropout'])
        self.head.load_state_dict(state); self.head.to(self.device).eval()

    @classmethod
    def from_pretrained(cls, path, device=None):
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path)
        cfg = json.load(open(f'{path}/config.json'))
        return cls(cfg, torch.load(f'{path}/head.pt', map_location='cpu'), device)

    @torch.no_grad()
    def probs(self, wav):
        wav = np.asarray(wav, np.float32)[:30 * 16000]
        f = self.fe(wav, sampling_rate=16000, return_tensors='pt').input_features.to(self.device, self.dt)
        T = max(1, int(np.ceil(len(wav) / 320)))                 # 50 encoder frames per second
        x = self.enc(f).last_hidden_state[:, :T].float()
        return self.head(x, torch.ones(1, T, device=self.device))[0].softmax(0).cpu().numpy()

    def predict(self, items):
        """items: paths (any sample rate, resampled to 16 kHz) or 16 kHz float waveforms."""
        out = []
        for it in items:
            wav = librosa.load(it, sr=16000)[0] if isinstance(it, str) else it
            out.append({c: float(v) for c, v in zip(self.classes, self.probs(wav))})
        return out

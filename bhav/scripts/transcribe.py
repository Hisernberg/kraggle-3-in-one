"""CPU transcription of all competition clips with a Hindi Whisper ASR model -> work/transcripts_<name>.csv."""
import sys, os, time
import numpy as np, pandas as pd, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, DATA, BH
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
name, repo = sys.argv[1], sys.argv[2]
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
m = load_meta()
proc = WhisperProcessor.from_pretrained(repo); model = WhisperForConditionalGeneration.from_pretrained(repo).eval()
if len(sys.argv) > 4:   # base model whose generation config to use (older fine-tunes ship an outdated one)
    model.generation_config = GenerationConfig.from_pretrained(sys.argv[4])
out, t = [], time.time()
B = 8
paths = [f'{DATA}/{s}/{f}' for s, f in zip(m.split, m.filename)]
with torch.no_grad():
    for i in range(0, len(paths), B):
        ws = [sf.read(p, dtype='float32')[0] for p in paths[i:i + B]]
        feats = proc(ws, sampling_rate=16000, return_tensors='pt').input_features
        ids = model.generate(feats, language='hi', task='transcribe', max_new_tokens=96)
        out += proc.batch_decode(ids, skip_special_tokens=True)
        if i % 80 == 0: print(i, f'{time.time()-t:.0f}s', out[-1][:60], flush=True)
m.assign(text=[o.strip() for o in out])[['Id', 'split', 'emotion', 'text']].to_csv(f'{BH}/work/transcripts_{name}.csv', index=False)
print('done', time.time() - t)

---
language:
- hi
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: audio-classification
base_model: openai/whisper-large-v3
tags:
- speech-emotion-recognition
- hindi
- whisper
- audio-classification
---

# BhavVaani Whisper SER: Hindi speech emotion recognition

Four-class speech emotion recognition for Hindi (**angry, happy, neutral, sad**), built for the Kaggle community
competition *DataVerse: Detecting Emotions from Hindi Speech* (SRCASW-BhavVaani data).

## Model

* Backbone: `openai/whisper-large-v3` encoder, **frozen**, run exactly as in pre-training (30 s padded log-mel input).
  Its final-layer output (after the last LayerNorm) on the frames that belong to the audio is the input of the head.
* Head (`head.pt`, 1.4 MB, the only trained part): LayerNorm → dropout → Linear 1280→256 → GELU → attentive
  statistics pooling (attention-weighted mean and standard deviation over frames) → dropout → Linear 512→4.
* Training: the 826 labelled SRCASW-BhavVaani training clips (16 kHz, 1-11 s); AdamW (lr 1e-3, weight decay 0.05),
  one-cycle schedule, 30 epochs, batch 32, label smoothing 0.1, random frame crops and time masks. No external data
  (public Hindi/Indian emotion corpora were tried and lowered validation scores).

## Usage

```python
# pip install torch transformers librosa huggingface_hub
from huggingface_hub import hf_hub_download
import importlib.util
path = hf_hub_download('Nabidnur/bhavvaani-whisper-ser', 'inference.py')
spec = importlib.util.spec_from_file_location('bhav', path); bhav = importlib.util.module_from_spec(spec); spec.loader.exec_module(bhav)
model = bhav.BhavWhisperSER.from_pretrained('Nabidnur/bhavvaani-whisper-ser')
print(model.predict(['speech.wav']))   # [{'angry': 0.01, 'happy': 0.02, 'neutral': 0.95, 'sad': 0.02}]
```

## Evaluation

Out-of-fold macro-F1 on the 826 training clips (3 repetitions of stratified 5-fold CV):

| Subset | Macro-F1 |
|---|---|
| clips without a duplicate in the training folds (566) | 0.887 |
| all clips, copying the label of a training duplicate where one exists | 0.921 |

The corpus contains re-saved copies of the same recording, so "all clips" is optimistic; the first row is the
honest number. For comparison, a logistic-regression probe on the same frozen features reaches 0.884, WavLM-large
0.787 and XLS-R 0.813. In the competition this head is one member of a blend of Whisper-family probes.

## Limitations

Trained on a small (826 clips), acted plus spontaneous, single-corpus dataset; expect lower accuracy on other
recording conditions, speakers and dialects. Four emotions only. Not for decisions about people.

## Data and citation

Trained on **SRCASW-BhavVaani**, the Hindi speech emotion dataset of the Kaggle competition
[DataVerse: Detecting Emotions from Hindi Speech](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech)
(ENIAC, Department of Computer Science, Shaheed Rajguru College of Applied Sciences for Women, University of Delhi).

The competition audio is not redistributed here (competition rules); only the model weights are released.

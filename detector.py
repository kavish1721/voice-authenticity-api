import os
import sys
from pathlib import Path

import torch

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AASIST_DIR = BASE_DIR / "aasist-source"
WAV2VEC_DIR = BASE_DIR / "model-source"
MODELS_DIR = BASE_DIR / "models"

# Make the two source repositories importable
sys.path.insert(0, str(AASIST_DIR))
sys.path.insert(0, str(WAV2VEC_DIR))

# ============================================================
# MODEL SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

AASIST_CHECKPOINT = MODELS_DIR / "weights" / "AASIST.pth"

# The large Wav2Vec2 checkpoint is NOT stored in Vercel.
# It is downloaded from Hugging Face when the function starts.
HF_REPO_ID = "Sara1708/deepfake-audio-wav2vec2"
HF_CHECKPOINT_NAME = "stage2_best.pt"

# Decision thresholds
SYNTHETIC_MODEL_THRESHOLD = 0.70
HUMAN_MODEL_THRESHOLD = 0.70

SYNTHETIC_COMBINED_THRESHOLD = 0.75
HUMAN_COMBINED_THRESHOLD = 0.25


# ============================================================
# GLOBAL MODELS
# ============================================================

aasist_model = None
wav2vec_detector = None


# ============================================================
# DOWNLOAD Wav2Vec2 CHECKPOINT
# ============================================================

def get_wav2vec_checkpoint():
    """
    Download the official stage2_best.pt checkpoint from
    Hugging Face if it is not already cached.

    The file is approximately 491 MB, so it is intentionally
    NOT included in the Vercel deployment package.
    """

    from huggingface_hub import hf_hub_download

    print("Checking Wav2Vec2 checkpoint...")

    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CHECKPOINT_NAME,
    )

    print(f"Wav2Vec2 checkpoint ready: {checkpoint_path}")

    return checkpoint_path


# ============================================================
# LOAD AASIST
# ============================================================

def load_aasist():
    global aasist_model

    print("Loading AASIST...")

    from models.AASIST import Model

    # Official AASIST configuration used by the project.
    config = {
        "architecture": {
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "blocks": [2, 2],
            "nb_fc_node": 160,
            "gru_node": 1024,
            "nb_gru_layer": 1,
            "nb_head": 1,
            "nb_mel": 128,
            "freq": 1024,
            "lstm_node": 128,
            "nb_lstm_layer": 2,
        }
    }

    aasist_model = Model(config).to(DEVICE)

    checkpoint = torch.load(
        AASIST_CHECKPOINT,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    aasist_model.load_state_dict(checkpoint, strict=False)
    aasist_model.eval()

    print("AASIST ready")


# ============================================================
# LOAD WAV2VEC2
# ============================================================

def load_wav2vec2():
    global wav2vec_detector

    print("Loading Wav2Vec2 detector...")

    checkpoint_path = get_wav2vec_checkpoint()

    from src.inference.predict import DeepfakeDetector

    wav2vec_detector = DeepfakeDetector(
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )

    print("Wav2Vec2 ready")


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_models():
    global aasist_model, wav2vec_detector

    if aasist_model is not None and wav2vec_detector is not None:
        return

    print("=" * 60)
    print("LOADING VOICE AUTHENTICITY MODELS")
    print("=" * 60)

    load_aasist()
    load_wav2vec2()

    print("=" * 60)
    print("ALL MODELS READY")
    print("=" * 60)


# ============================================================
# AASIST PREDICTION
# ============================================================

def predict_aasist(audio_path):
    """
    Run AASIST over multiple overlapping 4-second windows.
    """

    import soundfile as sf
    import torch.nn.functional as F

    waveform, sample_rate = sf.read(audio_path)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = torch.tensor(
        waveform,
        dtype=torch.float32,
    )

    # Resampling if necessary
    if sample_rate != 16000:
        import torchaudio

        waveform = waveform.unsqueeze(0)

        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            16000,
        )

        waveform = waveform.squeeze(0)

    target_length = 64600
    hop_length = 32300

    windows = []

    if waveform.numel() <= target_length:
        padded = F.pad(
            waveform,
            (0, target_length - waveform.numel()),
        )
        windows.append(padded)
    else:
        start = 0

        while start < waveform.numel():
            chunk = waveform[start:start + target_length]

            if chunk.numel() < target_length:
                chunk = F.pad(
                    chunk,
                    (0, target_length - chunk.numel()),
                )

            windows.append(chunk)

            if start + target_length >= waveform.numel():
                break

            start += hop_length

    spoof_probabilities = []

    with torch.no_grad():
        for chunk in windows:
            batch = chunk.unsqueeze(0).to(DEVICE)

            output = aasist_model(batch)

            # AASIST output can be either logits or a tuple.
            if isinstance(output, tuple):
                output = output[0]

            if output.ndim == 1:
                output = output.unsqueeze(0)

            probabilities = torch.softmax(output, dim=-1)

            # Official AASIST convention:
            # class 0 = bonafide
            # class 1 = spoof
            spoof_probability = float(
                probabilities[0, 1].item()
            )

            spoof_probabilities.append(spoof_probability)

    if not spoof_probabilities:
        spoof_probabilities = [0.5]

    spoof_probability = sum(spoof_probabilities) / len(
        spoof_probabilities
    )

    human_probability = 1.0 - spoof_probability

    return {
        "human_probability": human_probability,
        "synthetic_probability": spoof_probability,
        "windows": len(spoof_probabilities),
    }


# ============================================================
# WAV2VEC2 PREDICTION
# ============================================================

def predict_wav2vec2(audio_path):
    """
    Run the official Wav2Vec2 deepfake detector.
    """

    result = wav2vec_detector.predict(
        audio_path,
        return_per_window=True,
    )

    spoof_probability = float(
        result["spoof_probability"]
    )

    human_probability = 1.0 - spoof_probability

    windows = result.get("windows", [])

    if isinstance(windows, list):
        window_count = len(windows)
    else:
        window_count = 1

    return {
        "human_probability": human_probability,
        "synthetic_probability": spoof_probability,
        "windows": window_count,
    }


# ============================================================
# COMBINED ANALYSIS
# ============================================================

def analyze_voice(audio_path):
    """
    Run both detectors and combine their evidence.

    AASIST weight: 60%
    Wav2Vec2 weight: 40%

    If the models disagree strongly, return UNCERTAIN.
    """

    import soundfile as sf

    # Make sure models are loaded
    load_models()

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    try:
        info = sf.info(audio_path)
        duration = float(info.duration)
    except Exception:
        duration = 0.0

    print("=" * 60)
    print("RUNNING COMBINED DETECTOR")
    print("=" * 60)

    # --------------------------------------------------------
    # AASIST
    # --------------------------------------------------------

    aasist_result = predict_aasist(audio_path)

    # --------------------------------------------------------
    # Wav2Vec2
    # --------------------------------------------------------

    wav2vec_result = predict_wav2vec2(audio_path)

    # --------------------------------------------------------
    # Combined probability
    # --------------------------------------------------------

    aasist_spoof = aasist_result[
        "synthetic_probability"
    ]

    wav2vec_spoof = wav2vec_result[
        "synthetic_probability"
    ]

    combined_spoof = (
        0.60 * aasist_spoof
        + 0.40 * wav2vec_spoof
    )

    combined_human = 1.0 - combined_spoof

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if (
        aasist_spoof >= SYNTHETIC_MODEL_THRESHOLD
        and wav2vec_spoof >= SYNTHETIC_MODEL_THRESHOLD
        and combined_spoof >= SYNTHETIC_COMBINED_THRESHOLD
    ):
        prediction = "synthetic"

        confidence = combined_spoof

    elif (
        aasist_spoof <= (1.0 - HUMAN_MODEL_THRESHOLD)
        and wav2vec_spoof <= (1.0 - HUMAN_MODEL_THRESHOLD)
        and combined_human >= HUMAN_COMBINED_THRESHOLD
    ):
        prediction = "human"

        confidence = combined_human

    else:
        prediction = "uncertain"

        # For uncertainty, report the stronger combined evidence
        confidence = max(
            combined_human,
            combined_spoof,
        )

    # --------------------------------------------------------
    # Model agreement
    # --------------------------------------------------------

    model_difference = abs(
        aasist_spoof - wav2vec_spoof
    )

    if model_difference <= 0.20:
        model_agreement = "strong"

    elif model_difference <= 0.40:
        model_agreement = "moderate"

    else:
        model_agreement = "disagreement"

    result = {
        "prediction": prediction,
        "confidence": confidence,

        "human_probability": combined_human,
        "synthetic_probability": combined_spoof,

        "duration": duration,

        "aasist": aasist_result,

        "wav2vec2": wav2vec_result,

        "model_agreement": model_agreement,
    }

    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Prediction: {prediction}")
    print(f"Human probability: {combined_human:.4f}")
    print(f"Synthetic probability: {combined_spoof:.4f}")
    print(f"Model agreement: {model_agreement}")
    print("=" * 60)

    return result


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":
    load_models()
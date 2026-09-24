import numpy as np
import soundfile as sf
import librosa

FILES = {
    "Human 1": "../human-test.wav",
    "Human 2": "../human-test-2.wav",
}

WINDOW = 64000
HOP = 32000

for name, path in FILES.items():

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    total_windows = max(1, int(np.ceil((len(audio) - WINDOW) / HOP)) + 1)

    for i in range(total_windows):

        start = i * HOP
        end = start + WINDOW

        chunk = audio[start:end]

        if len(chunk) < WINDOW:
            chunk = np.pad(chunk, (0, WINDOW - len(chunk)))

        rms = np.sqrt(np.mean(chunk ** 2))
        peak = np.max(np.abs(chunk))
        silence = np.mean(np.abs(chunk) < 0.01)

        zcr = np.mean(
            librosa.feature.zero_crossing_rate(chunk)[0]
        )

        centroid = np.mean(
            librosa.feature.spectral_centroid(
                y=chunk,
                sr=sr
            )[0]
        )

        print(
            f"Window {i+1:02d} | "
            f"time {start/sr:5.1f}-{min(end,len(audio))/sr:5.1f}s | "
            f"RMS {rms:.4f} | "
            f"Peak {peak:.3f} | "
            f"Silence {silence*100:5.1f}% | "
            f"ZCR {zcr:.4f} | "
            f"Centroid {centroid:7.1f} Hz"
        )
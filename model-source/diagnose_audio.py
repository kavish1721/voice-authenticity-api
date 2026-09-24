import numpy as np
import soundfile as sf
import librosa

FILES = {
    "Human 1": "../human-test.wav",
    "Human 2": "../human-test-2.wav",
}

for name, path in FILES.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    audio, sr = sf.read(path)

    # Convert stereo to mono if necessary
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    duration = len(audio) / sr
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    # Silence threshold
    silence_threshold = 0.01
    silence_ratio = np.mean(np.abs(audio) < silence_threshold)

    # Clipping
    clipping_ratio = np.mean(np.abs(audio) >= 0.999)

    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])

    # Spectral features
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    )

    spectral_bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    )

    print(f"Sample rate       : {sr} Hz")
    print(f"Samples            : {len(audio)}")
    print(f"Duration            : {duration:.2f} sec")
    print(f"Peak amplitude      : {peak:.4f}")
    print(f"RMS                 : {rms:.4f}")
    print(f"Silence ratio       : {silence_ratio * 100:.2f}%")
    print(f"Clipping ratio      : {clipping_ratio * 100:.4f}%")
    print(f"Zero crossing rate  : {zcr:.5f}")
    print(f"Spectral centroid   : {spectral_centroid:.2f} Hz")
    print(f"Spectral bandwidth   : {spectral_bandwidth:.2f} Hz")

print("\nDiagnostic complete.")
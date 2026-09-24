import os
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from detector import load_models, analyze_voice


app = FastAPI(
    title="Voice Authenticity API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://voice-authenticity-ai.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("STARTING VOICE AUTHENTICITY API")
    print("=" * 60)

    load_models()

    print("API READY")
    print("=" * 60)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api")
async def root():
    return {
        "success": True,
        "message": "Voice Authenticity API is running",
    }


@app.get("/api/health")
async def health():
    return {
        "success": True,
        "status": "healthy",
    }


# ============================================================
# AUDIO ANALYSIS
# ============================================================

@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):

    temp_input = None
    temp_wav = None

    try:
        # ----------------------------------------------------
        # Save uploaded audio
        # ----------------------------------------------------

        suffix = os.path.splitext(
            audio.filename or ""
        )[1]

        if not suffix:
            suffix = ".webm"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_input = temp_file.name

            content = await audio.read()

            temp_file.write(content)

        # ----------------------------------------------------
        # Convert audio to 16 kHz mono WAV
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as wav_file:

            temp_wav = wav_file.name

        try:
            import imageio_ffmpeg

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg_path,
                "-y",
                "-i",
                temp_input,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                temp_wav,
            ]

            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        except Exception as exc:
            print("FFmpeg conversion failed:")
            print(exc)

            raise

        # ----------------------------------------------------
        # Run detector
        # ----------------------------------------------------

        print("=" * 60)
        print("RUNNING COMBINED DETECTOR")
        print("=" * 60)

        result = analyze_voice(temp_wav)

        print("=" * 60)
        print("ANALYSIS COMPLETE")
        print(result)
        print("=" * 60)

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "success": True,

            "prediction": result["prediction"],

            "confidence": result["confidence"],

            "human_probability": result[
                "human_probability"
            ],

            "synthetic_probability": result[
                "synthetic_probability"
            ],

            "duration": result["duration"],

            "models": {
                "aasist": {
                    "human_probability": result[
                        "aasist"
                    ]["human_probability"],

                    "synthetic_probability": result[
                        "aasist"
                    ]["synthetic_probability"],

                    "windows": result[
                        "aasist"
                    ]["windows"],
                },

                "wav2vec2": {
                    "human_probability": result[
                        "wav2vec2"
                    ]["human_probability"],

                    "synthetic_probability": result[
                        "wav2vec2"
                    ]["synthetic_probability"],

                    "windows": result[
                        "wav2vec2"
                    ]["windows"],
                },
            },

            "model_agreement": result.get(
                "model_agreement"
            ),
        }

    except Exception as exc:

        print("=" * 60)
        print("ANALYSIS ERROR")
        print("=" * 60)
        print(repr(exc))
        print("=" * 60)

        return {
            "success": False,
            "error": str(exc),
        }

    finally:

        # ----------------------------------------------------
        # Cleanup temporary files
        # ----------------------------------------------------

        for path in [temp_input, temp_wav]:

            if path and os.path.exists(path):

                try:
                    os.remove(path)

                except Exception:
                    pass
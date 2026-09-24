import os
import uuid
import subprocess
import tempfile

import imageio_ffmpeg

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from detector import load_models, analyze_voice


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Voice Authenticity Detector",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL STATUS
# ============================================================

models_loaded = False


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global models_loaded

    print("\n")
    print("=" * 60)
    print("STARTING VOICE AUTHENTICITY API")
    print("=" * 60)

    try:

        load_models()

        models_loaded = True

        print("\n")
        print("=" * 60)
        print("ALL MODELS READY")
        print("=" * 60)

        print("\nAPI READY")
        print("=" * 60)

    except Exception as e:

        models_loaded = False

        print("\nMODEL LOADING ERROR:")
        print(str(e))

        raise


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "service": "Voice Authenticity Detector",
        "models_loaded": models_loaded,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok" if models_loaded else "loading",
        "models_loaded": models_loaded,
    }


# ============================================================
# ANALYZE VOICE
# IMPORTANT:
# Frontend sends FormData field named "audio"
# ============================================================

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):

    # --------------------------------------------------------
    # Check model status
    # --------------------------------------------------------

    if not models_loaded:

        raise HTTPException(
            status_code=503,
            detail="Models are still loading."
        )

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if not audio.filename:

        raise HTTPException(
            status_code=400,
            detail="No audio file supplied."
        )

    print("\n")
    print("=" * 60)
    print("NEW AUDIO ANALYSIS REQUEST")
    print("=" * 60)

    print(f"Received file: {audio.filename}")

    # --------------------------------------------------------
    # Create unique temporary file names
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    input_path = os.path.join(
        tempfile.gettempdir(),
        f"voice_input_{unique_id}"
    )

    wav_path = os.path.join(
        tempfile.gettempdir(),
        f"voice_audio_{unique_id}.wav"
    )

    try:

        # ====================================================
        # READ UPLOADED AUDIO
        # ====================================================

        audio_bytes = await audio.read()

        if not audio_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded audio is empty."
            )

        print(
            f"Uploaded size: "
            f"{len(audio_bytes) / 1024:.2f} KB"
        )

        # ====================================================
        # SAVE ORIGINAL BROWSER AUDIO
        # ====================================================

        with open(input_path, "wb") as f:

            f.write(audio_bytes)

        print(
            f"Temporary input file created: "
            f"{input_path}"
        )

        # ====================================================
        # GET FFMPEG
        # ====================================================

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        # ====================================================
        # CONVERT WEBM/OPUS → WAV
        #
        # 16 kHz
        # Mono
        # PCM
        # ====================================================

        command = [
            ffmpeg,

            "-y",

            "-i",
            input_path,

            "-ar",
            "16000",

            "-ac",
            "1",

            "-sample_fmt",
            "s16",

            wav_path,
        ]

        print("\nConverting audio...")

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # ====================================================
        # CHECK FFMPEG
        # ====================================================

        if process.returncode != 0:

            print("\nFFMPEG ERROR:")
            print(process.stderr)

            raise HTTPException(
                status_code=400,
                detail="Could not decode the uploaded audio."
            )

        print("Audio converted to 16 kHz mono WAV.")

        # ====================================================
        # RUN COMBINED DETECTOR
        # ====================================================

        print("\n")
        print("=" * 60)
        print("RUNNING COMBINED DETECTOR")
        print("=" * 60)

        result = analyze_voice(wav_path)

        # ====================================================
        # PREPARE RESPONSE
        # ====================================================

        response = {

            "success": True,

            "prediction":
                result["prediction"],

            "confidence":
                result["confidence"],

            "human_probability":
                result["human_probability"],

            "synthetic_probability":
                result["synthetic_probability"],

            "duration":
                result["duration"],

            "models": {

                "aasist": {

                    "human_probability":
                        result["aasist"][
                            "human_probability"
                        ],

                    "synthetic_probability":
                        result["aasist"][
                            "synthetic_probability"
                        ],

                    "windows":
                        result["aasist"]["windows"],
                },

                "wav2vec2": {

                    "human_probability":
                        result["wav2vec2"][
                            "human_probability"
                        ],

                    "synthetic_probability":
                        result["wav2vec2"][
                            "synthetic_probability"
                        ],

                    "windows":
                        result["wav2vec2"]["windows"],
                },
            },
        }

        # ====================================================
        # PRINT FINAL RESULT
        # ====================================================

        print("\n")
        print("=" * 60)
        print("API FINAL RESULT")
        print("=" * 60)

        print(
            "Prediction:",
            response["prediction"]
        )

        print(
            "Human probability:",
            f"{response['human_probability'] * 100:.2f}%"
        )

        print(
            "Synthetic probability:",
            f"{response['synthetic_probability'] * 100:.2f}%"
        )

        print(
            "Confidence:",
            f"{response['confidence'] * 100:.2f}%"
        )

        print(
            "Duration:",
            f"{response['duration']:.2f} seconds"
        )

        print("=" * 60)

        return response

    # ========================================================
    # FASTAPI ERRORS
    # ========================================================

    except HTTPException:

        raise

    # ========================================================
    # UNEXPECTED ERRORS
    # ========================================================

    except Exception as e:

        print("\n")
        print("=" * 60)
        print("ANALYSIS ERROR")
        print("=" * 60)

        print(type(e).__name__)
        print(str(e))

        print("=" * 60)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        for path in [
            input_path,
            wav_path,
        ]:

            try:

                if os.path.exists(path):

                    os.remove(path)

                    print(
                        f"Deleted temporary file: {path}"
                    )

            except Exception as cleanup_error:

                print(
                    f"Cleanup warning: "
                    f"{cleanup_error}"
                )
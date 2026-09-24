from fastapi import FastAPI

app = FastAPI()

@app.get("/api")
def root():
    return {
        "success": True,
        "message": "FastAPI is working on Vercel"
    }

@app.get("/api/health")
def health():
    return {
        "success": True,
        "status": "healthy"
    }

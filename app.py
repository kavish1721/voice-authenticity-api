from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"success": True, "status": "healthy"}

from fastapi import FastAPI

from app.api.v1 import health, predictions, zones

app = FastAPI(
    title="Trail Coach API",
    description="Backend for the Trail Coach ultra-trail coaching platform.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(zones.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Trail Coach API is running."}

from fastapi import FastAPI

app = FastAPI(
    title="Trail Coach API",
    description="Backend for the Trail Coach ultra-trail coaching platform.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Trail Coach API is running."}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

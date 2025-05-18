from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create a simple FastAPI app
app = FastAPI(title="MCP Tool Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple addition endpoint
@app.get("/tool/add")
@app.post("/tool/add")
async def add_numbers(a: int, b: int):
    """Add two numbers."""
    return {"result": a + b}

# Simple greeting endpoint
@app.get("/greeting/{name}")
async def get_greeting(name: str):
    """Return a friendly greeting."""
    return {"result": f"Hello, {name}! Nice to meet you."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6790)  # runs on http://localhost:6790

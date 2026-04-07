"""
FastAPI server for PharmaSimEnvironment
Exposes reset() and step() as HTTP endpoints
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import PharmacistAction, PharmacistObservation
from environment import PharmaSimEnvironment

app = FastAPI(
    title="PharmaSimEnvironment API",
    description="Drug formulation validation environment for AI pharmacist training",
    version="1.0.0"
)

# Enable CORS for HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = PharmaSimEnvironment()


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "PharmaSimEnvironment is running",
        "version": "1.0.0",
        "tasks": 3,
        "description": "Drug formulation validation for AI pharmacist agents"
    }


@app.post("/reset", response_model=PharmacistObservation)
async def reset():
    """
    Start new episode
    Returns initial observation with patient case
    """
    try:
        observation = env.reset()
        return observation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(action: PharmacistAction):
    """
    Take pharmacist action, return observation + reward
    
    Args:
        action: PharmacistAction with decision and reasoning
        
    Returns:
        dict with observation, reward, done
    """
    try:
        observation, reward, done = env.step(action)
        
        return {
            "observation": observation,
            "reward": reward,
            "done": done
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def get_state():
    """Return current environment state (for debugging)"""
    try:
        state = env.get_state()
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")


def main() -> None:
    """Run the FastAPI app with uvicorn for local/dev execution."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


# For local testing
if __name__ == "__main__":
    main()

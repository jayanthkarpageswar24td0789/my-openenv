"""
HTTP client for PharmaSimEnvironment
Connects to server via REST API
"""

import os
import sys
from typing import Tuple

import httpx
from models import PharmacistAction, PharmacistObservation


class PharmaSimClient:
    """
    Client to interact with PharmaSimEnvironment server
    
    Usage:
        client = PharmaSimClient(base_url="http://localhost:8000")
        obs = client.reset()
        obs, reward, done = client.step(action)
        client.close()
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", verbose: bool = True):
        """
        Initialize client
        
        Args:
            base_url: Server URL (local or HF Spaces)
            verbose: Whether to print connection status messages
        """
        self.base_url = base_url.rstrip("/")
        self.verbose = verbose
        self.client = httpx.Client(timeout=30.0)
        self.connected = self._check_connection()
    
    def _check_connection(self) -> bool:
        """Verify server is reachable"""
        try:
            response = self.client.get(f"{self.base_url}/")
            response.raise_for_status()
            if self.verbose:
                print(f"✓ Connected to PharmaSimEnvironment: {response.json()['status']}")
            return True
        except Exception as e:
            raise ConnectionError(
                f"Could not reach PharmaSimEnvironment at {self.base_url}: {e}"
            ) from e

    def _serialize_action(self, action: PharmacistAction) -> dict:
        """Serialize an action for Pydantic v1/v2 compatibility."""
        if hasattr(action, "model_dump"):
            return action.model_dump()
        return action.dict()
    
    def reset(self) -> PharmacistObservation:
        """
        Start new episode
        
        Returns:
            PharmacistObservation with patient case
        """
        try:
            response = self.client.post(f"{self.base_url}/reset")
            response.raise_for_status()
            data = response.json()
            return PharmacistObservation(**data)
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Reset failed: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Reset error: {str(e)}")
    
    def step(self, action: PharmacistAction) -> Tuple[PharmacistObservation, float, bool]:
        """
        Take pharmacist action
        
        Args:
            action: PharmacistAction with decision
            
        Returns:
            (observation, reward, done)
        """
        try:
            response = self.client.post(
                f"{self.base_url}/step",
                json=self._serialize_action(action)
            )
            response.raise_for_status()
            data = response.json()
            
            obs = PharmacistObservation(**data["observation"])
            reward = data["reward"]
            done = data["done"]
            
            return obs, reward, done
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Step failed: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Step error: {str(e)}")
    
    def close(self):
        """Close HTTP client"""
        self.client.close()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        self.close()


# Test script
if __name__ == "__main__":
    print("=== PharmaSimClient Test ===\n")
    base_url = os.getenv("SERVER_URL", "http://localhost:8000")
    print(f"Target server: {base_url}")
    
    try:
        with PharmaSimClient(base_url=base_url) as client:
            # Reset
            obs = client.reset()
            print(f"Task {obs.task_number}:")
            print(f"Patient: {obs.patient.age}yo, {obs.patient.conditions}")
            print(f"Formula: {obs.proposed_formula.active_ingredients}")
            print(f"\n{obs.message}\n")
            
            # Take action
            action = PharmacistAction(
                decision="REJECT",
                reasoning="Lactose excipient may be problematic for diabetic patient. Recommend lactose-free formulation."
            )
            
            obs, reward, done = client.step(action)
            print(f"Action taken: {action.decision}")
            print(f"Reward: {reward:.2f}")
            print(f"Done: {done}\n")
            print(f"Feedback:\n{obs.message}")
    except ConnectionError as e:
        print(f"✗ Connection failed: {e}")
        print("   Start the server first with:")
        print("   uvicorn server.app:app --reload --host 127.0.0.1 --port 8000")
        sys.exit(1)

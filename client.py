"""
HTTP client for PharmaSimEnvironment
Connects to server via REST API
"""

import httpx
import time
import sys
from typing import Tuple
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
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client
        
        Args:
            base_url: Server URL (local or HF Spaces)
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)
    
    def reset(self) -> PharmacistObservation:
        """
        Start new episode - with retry logic
        
        Returns:
            PharmacistObservation with patient case
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.post(f"{self.base_url}/reset")
                response.raise_for_status()
                data = response.json()
                return PharmacistObservation(**data)
            except httpx.ConnectError as e:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    raise RuntimeError(f"Cannot connect to {self.base_url} after {max_retries} attempts")
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Reset failed: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise RuntimeError(f"Reset error: {str(e)}")
    
    def step(self, action: PharmacistAction) -> Tuple[PharmacistObservation, float, bool]:
        """
        Take pharmacist action - with retry logic
        
        Args:
            action: PharmacistAction with decision
            
        Returns:
            (observation, reward, done)
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/step",
                    json=action.model_dump()
                )
                response.raise_for_status()
                data = response.json()
                
                obs = PharmacistObservation(**data["observation"])
                reward = data["reward"]
                done = data["done"]
                
                return obs, reward, done
                
            except httpx.ConnectError as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise RuntimeError(f"Cannot connect to {self.base_url}")
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Step failed: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise RuntimeError(f"Step error: {str(e)}")
    
    def close(self):
        """Close HTTP client"""
        try:
            self.client.close()
        except:
            pass
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        self.close()


# Test script
if __name__ == "__main__":
    print("=== PharmaSimClient Test ===\n")
    
    try:
        with PharmaSimClient(base_url="http://localhost:8000") as client:
            # Reset
            obs = client.reset()
            print(f"Task {obs.task_number}:")
            print(f"Patient: {obs.patient.age}yo, {obs.patient.conditions}")
            print(f"Formula: {obs.proposed_formula.active_ingredients}")
            print(f"\n{obs.message}\n")
            
            # Take action
            action = PharmacistAction(
                decision="REJECT",
                reasoning="Checking for contraindications"
            )
            
            obs, reward, done = client.step(action)
            print(f"Action taken: {action.decision}")
            print(f"Reward: {reward:.2f}")
            print(f"Done: {done}\n")
            print(f"Feedback:\n{obs.message}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
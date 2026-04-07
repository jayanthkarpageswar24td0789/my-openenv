"""
PharmaSimEnvironment - Drug Formulation Validation Environment
Main game logic for pharmaceutical safety checking tasks
"""

import random
import json
from pathlib import Path
from typing import Tuple, Dict, Any

from models import (
    PharmacistAction,
    PharmacistObservation,
    PharmacistState,
    PatientCase,
    DrugFormula
)


class PharmaSimEnvironment:
    """
    OpenEnv environment for training AI pharmacist agents
    
    Tasks:
    1. Easy: Basic contraindication detection (excipients)
    2. Medium: Dosage optimization for organ impairment
    3. Hard: Multi-drug interaction analysis
    """
    
    def __init__(self):
        self.state: PharmacistState = None
        self.current_case: Dict[str, Any] = None
        
        # Load task cases from JSON
        self.task_cases = self._load_task_cases()
        
    def _load_task_cases(self) -> Dict[str, list]:
        """Load patient cases from JSON file"""
        # Try multiple possible paths for data directory
        possible_paths = [
            Path(__file__).parent / "data" / "task_cases.json",  # When run from root
            Path(__file__).parent.parent / "data" / "task_cases.json",  # When run from server/
            Path.cwd() / "data" / "task_cases.json",  # Current working directory
        ]
        
        cases_file = None
        for path in possible_paths:
            if path.exists():
                cases_file = path
                break
        
        if cases_file:
            with open(cases_file, 'r') as f:
                return json.load(f)
        else:
            # Fallback to hardcoded cases if file missing
            return self._get_default_cases()
    
    def _get_default_cases(self) -> Dict[str, list]:
        """Fallback hardcoded cases if JSON file not found"""
        return {
            "task_1_easy": [
                {
                    "id": "case_101",
                    "patient": {
                        "age": 65,
                        "conditions": ["Type 2 Diabetes"],
                        "current_medications": [],
                        "lab_values": {}
                    },
                    "formula": {
                        "active_ingredients": ["Metformin 500mg"],
                        "excipients": ["Lactose", "Starch"],
                        "frequency": "Twice daily",
                        "indication": "Blood sugar control"
                    },
                    "correct_action": "REJECT",
                    "correct_reason": "Lactose may be problematic for diabetic patients"
                }
            ],
            "task_2_medium": [
                {
                    "id": "case_201",
                    "patient": {
                        "age": 58,
                        "conditions": ["Chronic Kidney Disease Stage 3"],
                        "current_medications": [],
                        "lab_values": {"eGFR": "45", "Creatinine": "1.8"}
                    },
                    "formula": {
                        "active_ingredients": ["Ibuprofen 400mg"],
                        "excipients": ["Gelatin"],
                        "frequency": "3 times daily",
                        "indication": "Joint pain"
                    },
                    "correct_action": "MODIFY",
                    "correct_reason": "NSAIDs nephrotoxic in CKD - reduce dose or use alternative"
                }
            ],
            "task_3_hard": [
                {
                    "id": "case_301",
                    "patient": {
                        "age": 72,
                        "conditions": ["Hypertension", "Atrial Fibrillation"],
                        "current_medications": ["Warfarin 5mg", "Amlodipine 10mg"],
                        "lab_values": {"INR": "2.4"}
                    },
                    "formula": {
                        "active_ingredients": ["Ciprofloxacin 500mg"],
                        "excipients": ["Lactose"],
                        "frequency": "Twice daily",
                        "indication": "UTI"
                    },
                    "correct_action": "REJECT",
                    "correct_reason": "Cipro + Warfarin = bleeding risk"
                }
            ]
        }
    
    def reset(self) -> PharmacistObservation:
        """
        Start new episode - randomly select one of 3 tasks
        Returns initial observation
        """
        # Randomly pick task difficulty
        task_id = random.choice([1, 2, 3])
        
        # Select random case from chosen task
        if task_id == 1:
            task_key = "task_1_easy"
            task_name = "Basic Contraindication Check"
        elif task_id == 2:
            task_key = "task_2_medium"
            task_name = "Dosage Optimization"
        else:
            task_key = "task_3_hard"
            task_name = "Multi-Drug Interaction Analysis"
        
        self.current_case = random.choice(self.task_cases[task_key])
        
        # Initialize state
        self.state = PharmacistState(
            task_id=task_id,
            patient_data=self.current_case,
            correct_action=self.current_case["correct_action"],
            correct_reason=self.current_case.get("correct_reason", ""),
            score=0.0,
            episode_step=0
        )
        
        # Build observation (what agent sees)
        observation = PharmacistObservation(
            patient=PatientCase(**self.current_case["patient"]),
            proposed_formula=DrugFormula(**self.current_case["formula"]),
            task_number=task_id,
            done=False,
            message=f"Task {task_id}: {task_name}\n\nAnalyze this patient case and validate the proposed formula for safety and efficacy.",
            reward=0.0
        )
        
        return observation
    
    def step(self, action: PharmacistAction) -> Tuple[PharmacistObservation, float, bool]:
        """
        Process agent's pharmacist decision
        
        Args:
            action: PharmacistAction with decision and reasoning
            
        Returns:
            (observation, reward, done)
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Grade the action
        reward = self._grade_action(action)
        self.state.score = reward
        self.state.episode_step += 1
        
        # Episode ends after one decision (single-turn tasks)
        done = True
        
        # Build result message
        result_message = self._build_result_message(action, reward)
        
        # Return final observation
        observation = PharmacistObservation(
            patient=PatientCase(**self.current_case["patient"]),
            proposed_formula=DrugFormula(**self.current_case["formula"]),
            task_number=self.state.task_id,
            done=done,
            message=result_message,
            reward=reward
        )
        
        return observation, reward, done
    
    def _grade_action(self, action: PharmacistAction) -> float:
        """
        Grade pharmacist decision based on rubric
        
        Scoring:
        - 0.99: Perfect (correct decision + good reasoning)
        - 0.7: Good (correct decision, weak reasoning OR wrong decision but identifies issue)
        - 0.4: Partial (mentions relevant concern but wrong conclusion)
        - 0.01: Wrong (dangerous decision or no reasoning)
        """
        correct_decision = self.state.correct_action
        agent_decision = action.decision
        reasoning = action.reasoning.lower()
        
        # Perfect score: correct decision
        if agent_decision == correct_decision:
            # Check reasoning quality
            if self.state.task_id == 1:
                # Task 1: Should mention lactose or excipient
                if "lactose" in reasoning or "excipient" in reasoning:
                    return 0.99
                else:
                    return 0.7  # Right answer but weak reasoning
            
            elif self.state.task_id == 2:
                # Task 2: Should mention kidney/renal/dose/nsaid
                keywords = ["kidney", "renal", "ckd", "dose", "nsaid", "nephrotoxic"]
                if any(kw in reasoning for kw in keywords):
                    return 0.99
                else:
                    return 0.7
            
            else:  # Task 3
                # Task 3: Should mention interaction or specific drugs
                keywords = ["interaction", "warfarin", "ciprofloxacin", "bleeding", "inr"]
                if any(kw in reasoning for kw in keywords):
                    return 0.99
                else:
                    return 0.7
        
        # Partial credit: wrong decision but identifies safety concern
        else:
            # Check if reasoning shows awareness of issue
            if self.state.task_id == 1:
                if "lactose" in reasoning or "diabetic" in reasoning or "contraindic" in reasoning:
                    return 0.4  # Saw the issue but wrong action
            
            elif self.state.task_id == 2:
                if any(kw in reasoning for kw in ["kidney", "renal", "dose", "reduce"]):
                    return 0.4
            
            else:  # Task 3
                if any(kw in reasoning for kw in ["interaction", "warfarin", "monitor"]):
                    return 0.4
            
            # No relevant reasoning - complete miss
            return 0.01
    
    def _build_result_message(self, action: PharmacistAction, reward: float) -> str:
        """Build feedback message for agent"""
        msg = f"Your Decision: {action.decision}\n"
        msg += f"Your Reasoning: {action.reasoning}\n\n"
        msg += f"--- Evaluation ---\n"
        msg += f"Expected Decision: {self.state.correct_action}\n"
        msg += f"Explanation: {self.state.correct_reason}\n\n"
        msg += f"Score: {reward:.2f}/1.00\n"
        
        if reward >= 0.9:
            msg += "✓ Excellent clinical decision!"
        elif reward >= 0.6:
            msg += "✓ Good catch, but reasoning could be stronger"
        elif reward >= 0.3:
            msg += "⚠ Partial credit - you identified concerns but wrong action"
        else:
            msg += "✗ Missed critical safety issue"
        
        return msg
    
    def get_state(self) -> PharmacistState:
        """Return current internal state (for debugging)"""
        return self.state


# Quick test
if __name__ == "__main__":
    env = PharmaSimEnvironment()
    
    print("=== PharmaSimEnvironment Test ===\n")
    
    # Test reset
    obs = env.reset()
    print(f"Task {obs.task_number}: {obs.message}\n")
    print(f"Patient: {obs.patient.age}yo, Conditions: {obs.patient.conditions}")
    print(f"Formula: {obs.proposed_formula.active_ingredients}")
    print(f"Excipients: {obs.proposed_formula.excipients}\n")
    
    # Test step
    action = PharmacistAction(
        decision="REJECT",
        reasoning="Lactose in excipients may be contraindicated for diabetic patient",
        warnings=["Check for lactose-free formulation"]
    )
    
    obs, reward, done = env.step(action)
    print(f"\nResult:\n{obs.message}")
    print(f"\nReward: {reward}, Done: {done}")

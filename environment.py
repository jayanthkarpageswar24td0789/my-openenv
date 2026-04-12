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

    def _determine_risk_level(self) -> str:
        """Classify the current case into a simple clinical risk tier."""
        patient = self.current_case.get("patient", {}) if self.current_case else {}
        medications = [med.lower() for med in patient.get("current_medications", [])]
        conditions = [condition.lower() for condition in patient.get("conditions", [])]

        if any("warfarin" in medication for medication in medications):
            return "HIGH"

        if any("kidney_disease" in condition or "ckd" in condition or "kidney" in condition for condition in conditions):
            return "MEDIUM"

        return "LOW"

    def _build_observation(self, done: bool, message: str, reward: float) -> PharmacistObservation:
        """Create an observation for the current case with derived risk metadata."""
        return PharmacistObservation(
            patient=PatientCase(**self.current_case["patient"]),
            proposed_formula=DrugFormula(**self.current_case["formula"]),
            task_number=self.state.task_id,
            done=done,
            message=message,
            reward=reward,
            risk_level=self._determine_risk_level(),
        )
    
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
        message = (
            f"Task {task_id}: {task_name}\n\n"
            "Analyze this patient case and validate the proposed formula for safety and efficacy."
        )
        return self._build_observation(done=False, message=message, reward=0.0)
    
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
        
        self.state.episode_step += 1

        if self.state.episode_step == 1:
            reward = 0.3
            self.state.score = reward
            message = (
                "Step 1 complete: partial evaluation recorded. "
                "Submit the second step for final grading."
            )
            observation = self._build_observation(done=False, message=message, reward=reward)
            return observation, reward, False

        reward = self.evaluate_action(action)
        self.state.score = reward

        result_message = self._build_result_message(action, reward)
        observation = self._build_observation(done=True, message=result_message, reward=reward)

        return observation, reward, True
    
    def evaluate_action(self, action: PharmacistAction) -> float:
        """
        Grade pharmacist decision using a deterministic rule-based rubric.
        """
        score = 0.0
        reasoning = (action.reasoning or "").lower()

        if action.decision == self.state.correct_action:
            score += 0.6
        else:
            score -= 0.2

        if action.reasoning and len(action.reasoning) > 10:
            score += 0.3

        if action.suggested_changes:
            score += 0.2

        if "bleeding" in reasoning or "interaction" in reasoning:
            score += 0.2

        # Ensure score strictly between (0, 1)
        score = max(0.01, min(score, 0.99))
        return score

    def _grade_action(self, action: PharmacistAction) -> float:
        """Backward-compatible wrapper for older callers."""
        return self.evaluate_action(action)
    
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

"""
PharmaSimEnvironment - Drug Formulation Validation Environment
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

# 🔥 GLOBAL SAFE CLAMP (CRITICAL FOR PHASE 2)
def clamp_score(score: float) -> float:
    try:
        score = float(score)
    except:
        return 0.5

    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return score


class PharmaSimEnvironment:

    def __init__(self):
        self.state: PharmacistState = None
        self.current_case: Dict[str, Any] = None
        self.task_cases = self._load_task_cases()

    def _load_task_cases(self) -> Dict[str, list]:
        possible_paths = [
            Path(__file__).parent / "data" / "task_cases.json",
            Path(__file__).parent.parent / "data" / "task_cases.json",
            Path.cwd() / "data" / "task_cases.json",
        ]

        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)

        return self._get_default_cases()

    def _get_default_cases(self):
        return {
            "task_1_easy": [{
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
                "correct_reason": "Lactose may be problematic"
            }],
            "task_2_medium": [{
                "id": "case_201",
                "patient": {
                    "age": 58,
                    "conditions": ["Chronic Kidney Disease"],
                    "current_medications": [],
                    "lab_values": {"eGFR": "45"}
                },
                "formula": {
                    "active_ingredients": ["Ibuprofen 400mg"],
                    "excipients": ["Gelatin"],
                    "frequency": "3 times daily"
                },
                "correct_action": "MODIFY",
                "correct_reason": "NSAIDs harmful in CKD"
            }],
            "task_3_hard": [{
                "id": "case_301",
                "patient": {
                    "age": 72,
                    "conditions": ["Hypertension"],
                    "current_medications": ["Warfarin"],
                    "lab_values": {"INR": "2.4"}
                },
                "formula": {
                    "active_ingredients": ["Ciprofloxacin"],
                    "excipients": ["Lactose"],
                    "frequency": "Twice daily"
                },
                "correct_action": "REJECT",
                "correct_reason": "Drug interaction risk"
            }]
        }

    def reset(self) -> PharmacistObservation:

        task_id = random.choice([1, 2, 3])

        if task_id == 1:
            task_key = "task_1_easy"
        elif task_id == 2:
            task_key = "task_2_medium"
        else:
            task_key = "task_3_hard"

        self.current_case = random.choice(self.task_cases[task_key])

        self.state = PharmacistState(
            task_id=task_id,
            patient_data=self.current_case,
            correct_action=self.current_case["correct_action"],
            correct_reason=self.current_case.get("correct_reason", ""),
            score=0.5,
            episode_step=0
        )

        return PharmacistObservation(
            patient=PatientCase(**self.current_case["patient"]),
            proposed_formula=DrugFormula(**self.current_case["formula"]),
            task_number=task_id,
            done=False,
            message="Analyze this patient case carefully",
            reward=clamp_score(0.01)  # ✅ NEVER ZERO
        )

    def step(self, action: PharmacistAction) -> Tuple[PharmacistObservation, float, bool]:

        if self.state is None:
            raise RuntimeError("Call reset first")

        self.state.episode_step += 1

        raw_score = self._grade_action(action)

        # 🔥 CRITICAL FIX
        reward = clamp_score(raw_score)

        self.state.score = reward

        observation = PharmacistObservation(
            patient=PatientCase(**self.current_case["patient"]),
            proposed_formula=DrugFormula(**self.current_case["formula"]),
            task_number=self.state.task_id,
            done=True,
            message=self._build_result_message(action, reward),
            reward=reward
        )

        return observation, reward, True

    def _grade_action(self, action: PharmacistAction) -> float:

        correct = self.state.correct_action
        decision = action.decision
        reasoning = (action.reasoning or "").lower()

        if decision == correct:
            if len(reasoning) > 10:
                return 0.99
            return 0.7

        if any(k in reasoning for k in ["risk", "interaction", "kidney", "bleeding"]):
            return 0.4

        return 0.01

    def _build_result_message(self, action, reward):
        return f"Decision: {action.decision}\nScore: {reward}"

    def get_state(self):
        return self.state


if __name__ == "__main__":
    env = PharmaSimEnvironment()

    obs = env.reset()
    print("RESET OK")

    action = PharmacistAction(
        decision="REJECT",
        reasoning="drug interaction risk"
    )

    obs, reward, done = env.step(action)
    print("STEP OK:", reward)
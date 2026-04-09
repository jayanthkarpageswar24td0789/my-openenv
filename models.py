"""
Pydantic models for PharmaSimEnvironment
Defines data structures for patient cases, drug formulas, actions, and observations
"""

from typing import List, Optional, Dict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pydantic import BaseModel, Field


# Base classes for OpenEnv compatibility
class Action(BaseModel):
    """Base action class"""
    pass


class Observation(BaseModel):
    """Base observation class"""
    pass


class State(BaseModel):
    """Base state class"""
    pass


class PatientCase(BaseModel):
    """Patient demographic and medical information"""
    age: int = Field(..., description="Patient age in years")
    conditions: List[str] = Field(default_factory=list, description="List of medical conditions")
    current_medications: List[str] = Field(default_factory=list, description="Current medications")
    lab_values: Dict[str, str] = Field(default_factory=dict, description="Laboratory test results")


class DrugFormula(BaseModel):
    """Proposed medication formulation"""
    active_ingredients: List[str] = Field(..., description="Active pharmaceutical ingredients with dosages")
    excipients: List[str] = Field(default_factory=list, description="Inactive ingredients (binders, fillers)")
    frequency: str = Field(..., description="Dosing frequency (e.g., 'twice daily')")
    indication: Optional[str] = Field(None, description="Reason for prescription")


class PharmacistAction(Action):
    """
    Action space for AI pharmacist agent
    Agent must make a decision and provide reasoning
    """
    decision: Literal["APPROVE", "REJECT", "MODIFY", "REQUEST_INFO"] = Field(
        ..., 
        description="Pharmacist's decision on the proposed formula"
    )
    reasoning: str = Field(..., description="Clinical reasoning for the decision")
    suggested_changes: Optional[str] = Field(None, description="Specific modifications if decision is MODIFY")
    warnings: Optional[List[str]] = Field(default_factory=list, description="Safety warnings to flag")


class PharmacistObservation(Observation):
    """
    Observation space - what the AI agent sees
    Contains patient data and proposed formula
    """
    patient: PatientCase = Field(..., description="Patient information")
    proposed_formula: DrugFormula = Field(..., description="Medication formula to validate")
    task_number: int = Field(..., description="Task difficulty level (1=easy, 2=medium, 3=hard)")
    done: bool = Field(default=False, description="Whether episode has ended")
    message: str = Field(default="", description="Instruction or feedback message")
    reward: float = Field(default=0.0, description="Reward signal for RL training")
    risk_level: str = Field(default="LOW", description="Clinical risk classification")


class PharmacistState(State):
    """
    Internal environment state (not visible to agent)
    Used for grading and episode management
    """
    task_id: int = Field(..., description="Current task number")
    patient_data: dict = Field(..., description="Full case data including correct answer")
    correct_action: str = Field(..., description="Expected decision (for grading)")
    correct_reason: str = Field(default="", description="Explanation of correct answer")
    score: float = Field(default=0.0, description="Episode cumulative score")
    episode_step: int = Field(default=0, description="Step counter within episode")

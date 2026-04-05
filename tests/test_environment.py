"""
Unit tests for PharmaSimEnvironment
Run with: pytest tests/test_environment.py
"""

import pytest
from environment import PharmaSimEnvironment
from models import PharmacistAction


def test_environment_reset():
    """Test environment initialization"""
    env = PharmaSimEnvironment()
    obs = env.reset()
    
    assert obs.task_number in [1, 2, 3]
    assert obs.done == False
    assert obs.patient.age > 0
    assert len(obs.proposed_formula.active_ingredients) > 0


def test_environment_step():
    """Test step function"""
    env = PharmaSimEnvironment()
    obs = env.reset()
    
    action = PharmacistAction(
        decision="REJECT",
        reasoning="Testing step function"
    )
    
    obs, reward, done = env.step(action)
    
    assert done == True
    assert 0.0 <= reward <= 1.0
    assert obs.reward == reward


def test_grading_perfect_answer():
    """Test perfect score scenario"""
    env = PharmaSimEnvironment()
    
    # Force Task 1 (contraindication)
    env.state = None
    obs = env.reset()
    
    # If task is contraindication task, give perfect answer
    if env.state.correct_action == "REJECT":
        action = PharmacistAction(
            decision="REJECT",
            reasoning="Lactose excipient contraindicated for diabetic patient"
        )
        obs, reward, done = env.step(action)
        
        # Should get high score for mentioning lactose
        assert reward >= 0.7


def test_grading_wrong_answer():
    """Test zero score for dangerous decision"""
    env = PharmaSimEnvironment()
    obs = env.reset()
    
    # Always approve (likely wrong for some tasks)
    action = PharmacistAction(
        decision="APPROVE",
        reasoning="No issues found"
    )
    
    obs, reward, done = env.step(action)
    
    # Should get 0 if correct answer was REJECT
    if env.state.correct_action != "APPROVE":
        assert reward <= 0.5


def test_multiple_episodes():
    """Test running multiple episodes"""
    env = PharmaSimEnvironment()
    
    for _ in range(5):
        obs = env.reset()
        
        action = PharmacistAction(
            decision="MODIFY",
            reasoning="Suggesting dose adjustment"
        )
        
        obs, reward, done = env.step(action)
        
        assert done == True
        assert 0.0 <= reward <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

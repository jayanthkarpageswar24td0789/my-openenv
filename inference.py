"""
Baseline inference script for PharmaSimEnvironment
Uses LLM (GPT-4 or Claude) to act as AI pharmacist
Required for hackathon validation - OUTPUTS STRUCTURED FORMAT
"""

import os
import sys
from openai import OpenAI
from client import PharmaSimClient
from models import PharmacistAction


# Environment variables (hackathon requirement)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

# Initialize OpenAI client
try:
    client_llm = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy-key-for-testing")
    )
except Exception as e:
    client_llm = None


def build_prompt(obs) -> str:
    """Build prompt for LLM pharmacist"""
    prompt = f"""You are a clinical pharmacist reviewing a prescription for safety.

**Patient Information:**
- Age: {obs.patient.age} years
- Medical Conditions: {', '.join(obs.patient.conditions) if obs.patient.conditions else 'None'}
- Current Medications: {', '.join(obs.patient.current_medications) if obs.patient.current_medications else 'None'}
- Lab Values: {', '.join([f'{k}: {v}' for k, v in obs.patient.lab_values.items()]) if obs.patient.lab_values else 'None'}

**Proposed Medication:**
- Active Ingredients: {', '.join(obs.proposed_formula.active_ingredients)}
- Excipients (inactive ingredients): {', '.join(obs.proposed_formula.excipients)}
- Dosing Frequency: {obs.proposed_formula.frequency}

**Your Task:**
Evaluate this prescription for:
1. Contraindications (are any ingredients unsafe for this patient?)
2. Drug interactions (will this interact with current medications?)
3. Dosage appropriateness (is the dose safe given patient's age/kidney function/etc?)

**Respond in this EXACT format:**
DECISION: [Choose ONLY one: APPROVE / REJECT / MODIFY]
REASONING: [1-2 sentences explaining your clinical rationale]

Be concise and specific. Focus on patient safety.
"""
    return prompt


def parse_llm_response(llm_output: str) -> PharmacistAction:
    """Parse LLM output into PharmacistAction"""
    lines = llm_output.strip().split('\n')
    
    decision = "REQUEST_INFO"  # Default fallback
    reasoning = llm_output  # Fallback to full output
    
    for line in lines:
        if line.startswith("DECISION:"):
            decision_text = line.replace("DECISION:", "").strip().upper()
            # Extract first word
            for keyword in ["APPROVE", "REJECT", "MODIFY", "REQUEST_INFO"]:
                if keyword in decision_text:
                    decision = keyword
                    break
        
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
    
    # If reasoning still empty, use full output
    if not reasoning or reasoning == llm_output:
        reasoning = llm_output[:200]  # Truncate if too long
    
    return PharmacistAction(
        decision=decision,
        reasoning=reasoning
    )


def run_single_episode(env_client, episode_num: int) -> float:
    """
    Run one episode with LLM agent
    PRINTS STRUCTURED OUTPUT FOR META VALIDATION
    
    Args:
        env_client: PharmaSimClient instance
        episode_num: Episode number for logging
        
    Returns:
        reward (float)
    """
    # Reset environment
    obs = env_client.reset()
    task_name = f"pharma_task_{obs.task_number}"
    
    # ✅ CRITICAL: Print [START] in exact format Meta expects
    print(f"[START] task={task_name} episode={episode_num}", flush=True)
    
    # Build prompt
    prompt = build_prompt(obs)
    
    # Get LLM response or use fallback
    if client_llm is None:
        # Fallback: Use rule-based decision if no LLM
        import random
        action = PharmacistAction(
            decision=random.choice(["APPROVE", "REJECT", "MODIFY"]),
            reasoning="Automated decision (no LLM configured)"
        )
    else:
        try:
            response = client_llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            
            llm_output = response.choices[0].message.content
            action = parse_llm_response(llm_output)
            
        except Exception as e:
            # Fallback on error
            action = PharmacistAction(
                decision="REQUEST_INFO",
                reasoning=f"LLM error: {str(e)[:100]}"
            )
    
    # ✅ CRITICAL: Print [STEP] in exact format
    print(f"[STEP] step=1 action={action.decision} reward=0.0 done=false error=null", flush=True)
    
    # Step environment
    obs, reward, done = env_client.step(action)
    
    # ✅ CRITICAL: Print [END] in exact format
    print(f"[END] task={task_name} score={reward:.2f} steps=1", flush=True)
    
    return reward


def main():
    """
    Main function - runs baseline evaluation
    STRUCTURED OUTPUT FORMAT FOR META VALIDATION
    """
    # Connect to environment
    env_client = PharmaSimClient(base_url=SERVER_URL)
    
    # Run 3 episodes (minimum for validation)
    num_episodes = 3
    scores = []
    
    try:
        for i in range(num_episodes):
            reward = run_single_episode(env_client, episode_num=i+1)
            scores.append(reward)
        
        # Print summary (optional, not required by Meta)
        avg_score = sum(scores) / len(scores)
        print(f"\n# Baseline Summary: {num_episodes} episodes, avg_score={avg_score:.3f}", flush=True)
        
    finally:
        env_client.close()


if __name__ == "__main__":
    main()
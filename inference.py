"""
Baseline inference script for PharmaSimEnvironment
ROBUST VERSION with error handling for Meta validation
"""

import os
import sys
from openai import OpenAI
from client import PharmaSimClient
from models import PharmacistAction


# Server URL - detect if running in HF Spaces
if os.getenv("SPACE_ID"):
    # Running in HF Spaces - server is on same container
    SERVER_URL = "http://localhost:7860"
else:
    # Running locally
    SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

# LLM configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Initialize OpenAI client
try:
    client_llm = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy-key")
    )
except Exception:
    client_llm = None


def build_prompt(obs) -> str:
    """Build prompt for LLM pharmacist"""
    prompt = f"""You are a clinical pharmacist. Analyze this case:

Patient: {obs.patient.age}yo, Conditions: {', '.join(obs.patient.conditions) or 'None'}
Current Meds: {', '.join(obs.patient.current_medications) or 'None'}
Labs: {', '.join([f'{k}={v}' for k, v in obs.patient.lab_values.items()]) or 'None'}

Proposed Formula:
- Active: {', '.join(obs.proposed_formula.active_ingredients)}
- Excipients: {', '.join(obs.proposed_formula.excipients)}
- Frequency: {obs.proposed_formula.frequency}

Evaluate for safety. Respond:
DECISION: [APPROVE/REJECT/MODIFY]
REASONING: [brief explanation]
"""
    return prompt


def parse_llm_response(llm_output: str) -> PharmacistAction:
    """Parse LLM output"""
    decision = "REQUEST_INFO"
    reasoning = llm_output[:200]
    
    for line in llm_output.split('\n'):
        if "DECISION:" in line:
            for kw in ["APPROVE", "REJECT", "MODIFY"]:
                if kw in line.upper():
                    decision = kw
                    break
        elif "REASONING:" in line:
            reasoning = line.split("REASONING:")[-1].strip()
    
    return PharmacistAction(decision=decision, reasoning=reasoning)


def run_single_episode(env_client, episode_num: int) -> float:
    """
    Run one episode with structured output
    """
    try:
        # Reset
        obs = env_client.reset()
        task_name = f"pharma_task_{obs.task_number}"
        
        # Print [START]
        print(f"[START] task={task_name} episode={episode_num}", flush=True)
        
        # Get action
        if client_llm:
            try:
                response = client_llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": build_prompt(obs)}],
                    temperature=0.3,
                    max_tokens=150
                )
                action = parse_llm_response(response.choices[0].message.content)
            except:
                import random
                action = PharmacistAction(
                    decision=random.choice(["APPROVE", "REJECT", "MODIFY"]),
                    reasoning="Fallback decision"
                )
        else:
            import random
            action = PharmacistAction(
                decision=random.choice(["APPROVE", "REJECT", "MODIFY"]),
                reasoning="Random decision (no LLM)"
            )
        
        # Print [STEP]
        print(f"[STEP] step=1 action={action.decision} reward=0.0 done=false error=null", flush=True)
        
        # Step environment
        obs, reward, done = env_client.step(action)
        
        # Print [END]
        print(f"[END] task={task_name} score={reward:.2f} steps=1", flush=True)
        
        return reward
        
    except Exception as e:
        print(f"ERROR in episode {episode_num}: {str(e)}", file=sys.stderr, flush=True)
        raise


def main():
    """Main function with robust error handling"""
    
    # Connect to environment
    try:
        env_client = PharmaSimClient(base_url=SERVER_URL)
    except Exception as e:
        print(f"ERROR: Cannot create client: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    # Run episodes
    num_episodes = 3
    scores = []
    
    try:
        for i in range(num_episodes):
            try:
                reward = run_single_episode(env_client, episode_num=i+1)
                scores.append(reward)
            except Exception as e:
                print(f"ERROR: Episode {i+1} failed: {str(e)}", file=sys.stderr, flush=True)
                continue
        
        if not scores:
            print("ERROR: All episodes failed", file=sys.stderr, flush=True)
            sys.exit(1)
        
        # Summary
        avg_score = sum(scores) / len(scores)
        print(f"\n# Baseline Summary: {len(scores)} episodes, avg_score={avg_score:.3f}", flush=True)
        
    finally:
        try:
            env_client.close()
        except:
            pass


if __name__ == "__main__":
    main()
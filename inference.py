"""
Baseline inference script for PharmaSimEnvironment
Uses LLM (GPT-4 or Claude) to act as AI pharmacist
Required for hackathon validation
"""

import os
import sys
import subprocess
import time
from urllib.parse import urlparse
from openai import OpenAI
from client import PharmaSimClient
from models import PharmacistAction


# Environment variables (hackathon requirement)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

TASK_NAMES = {
    1: "Basic Contraindication Check",
    2: "Dosage Optimization",
    3: "Multi-Drug Interaction Analysis",
}


def format_task_name(task_number: int) -> str:
    return TASK_NAMES.get(task_number, f"Task {task_number}")


def compact_text(value: str, limit: int = 160) -> str:
    text = " ".join(value.strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _is_local_server(server_url: str) -> bool:
    parsed = urlparse(server_url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0"}


def _start_local_server_if_needed(server_url: str):
    """Start local uvicorn server and wait until health check responds."""
    if not _is_local_server(server_url):
        return None

    parsed = urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to ~8 seconds for server startup.
    for _ in range(16):
        time.sleep(0.5)
        if process.poll() is not None:
            return None
        try:
            probe = PharmaSimClient(base_url=server_url, verbose=False)
            probe.close()
            return process
        except ConnectionError:
            continue

    process.terminate()
    return None

# Initialize OpenAI client
try:
    client_llm = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "")
    )
except Exception as e:
    print(f"Warning: LLM client initialization failed: {e}")
    print("Set OPENAI_API_KEY or HF_TOKEN environment variable")
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
- Indication: {obs.proposed_formula.indication or 'Not specified'}

**Your Task:**
Evaluate this prescription for:
1. Contraindications (are any ingredients unsafe for this patient?)
2. Drug interactions (will this interact with current medications?)
3. Dosage appropriateness (is the dose safe given patient's age/kidney function/etc?)

**Respond in this EXACT format:**
DECISION: [Choose ONLY one: APPROVE / REJECT / MODIFY / REQUEST_INFO]
REASONING: [1-2 sentences explaining your clinical rationale]

Be concise and specific. Focus on patient safety.
"""
    return prompt


def call_llm(prompt: str):
    """Call the configured LLM and return the raw response text."""
    if client_llm is None:
        return None

    response = client_llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content


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


def build_fallback_action(obs) -> PharmacistAction:
    """Use simple clinical rules when the LLM path is unavailable or invalid."""
    conditions = " ".join(obs.patient.conditions).lower()
    medications = " ".join(obs.patient.current_medications).lower()
    active_ingredients = " ".join(obs.proposed_formula.active_ingredients).lower()
    excipients = " ".join(obs.proposed_formula.excipients).lower()
    lab_values = " ".join(f"{key}: {value}" for key, value in obs.patient.lab_values.items()).lower()

    if obs.task_number == 1:
        if "diabetes" in conditions and "lactose" in excipients:
            return PharmacistAction(
                decision="REJECT",
                reasoning="Lactose excipient may be problematic for this diabetic patient. Recommend a lactose-free formulation."
            )
        return PharmacistAction(
            decision="APPROVE",
            reasoning="No contraindication detected. This appears to be an appropriate formulation for the patient."
        )

    if obs.task_number == 2:
        if "ibuprofen" in active_ingredients or "nsaid" in active_ingredients:
            return PharmacistAction(
                decision="MODIFY",
                reasoning="NSAIDs are nephrotoxic in CKD. Reduce the dose or switch to a safer alternative such as acetaminophen."
            )
        if "metformin" in active_ingredients and ("egfr" in lab_values or "ckd" in conditions):
            return PharmacistAction(
                decision="MODIFY",
                reasoning="Metformin dose should be reduced for reduced renal function, and a lactose-free formulation should be considered."
            )
        return PharmacistAction(
            decision="APPROVE",
            reasoning="No major renal dosing issue detected."
        )

    if obs.task_number == 3:
        if "warfarin" in medications and "ciprofloxacin" in active_ingredients:
            return PharmacistAction(
                decision="REJECT",
                reasoning="Critical interaction between ciprofloxacin and warfarin increases bleeding risk. Suggest nitrofurantoin or fosfomycin and monitor INR if needed."
            )
        if "spironolactone" in active_ingredients and ("ckd" in conditions or "potassium" in lab_values):
            return PharmacistAction(
                decision="REJECT",
                reasoning="Spironolactone in CKD with elevated potassium creates a high hyperkalemia risk. This should not be started without close monitoring."
            )
        return PharmacistAction(
            decision="APPROVE",
            reasoning="No critical interaction identified for the current formulation."
        )

    return PharmacistAction(
        decision="REQUEST_INFO",
        reasoning="Unable to determine a safe decision from the available information."
    )


def run_single_episode(env_client, task_num: int = None) -> float:
    """
    Run one episode with LLM agent
    MUST print in [START]/[STEP]/[END] format for validation
    """
    
    # Reset environment
    obs = env_client.reset()
    task_name = format_task_name(obs.task_number)
    
    # Print START block
    print(f"[START] task={task_name}", flush=True)

    # Build prompt
    prompt = build_prompt(obs)

    try:
        llm_output = call_llm(prompt)
        if llm_output is None:
            action = build_fallback_action(obs)
            llm_output = f"DECISION: {action.decision}. REASONING: {action.reasoning}"
        else:
            action = parse_llm_response(llm_output)
            if action.decision == "REQUEST_INFO":
                action = build_fallback_action(obs)
    except Exception:
        action = build_fallback_action(obs)
        llm_output = f"DECISION: {action.decision}. REASONING: {action.reasoning}"

    # Step environment
    obs, reward, done = env_client.step(action)
    
    # Print STEP block with step number and reward
    print(f"[STEP] step=1 reward={reward:.2f}", flush=True)
    # Print END block with task name, score, and number of steps
    print(f"[END] task={task_name} score={reward:.2f} steps=1", flush=True)

    return reward


def run_baseline(num_episodes: int = 10):
    """
    Run baseline evaluation across all 3 tasks
    
    Args:
        num_episodes: Number of episodes to run
    """
    print("=" * 60)
    print("PHARMASIM BASELINE EVALUATION")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    # Connect to environment
    env_client = None
    server_process = None
    try:
        env_client = PharmaSimClient(base_url=SERVER_URL, verbose=False)
    except ConnectionError as e:
        print(f"\n⚠ Connection failed: {e}")
        print("   Attempting to auto-start local server...")

        server_process = _start_local_server_if_needed(SERVER_URL)
        if server_process is None:
            print("\n✗ Could not auto-start server.")
            print("   Start the server first with:")
            print("   uvicorn server.app:app --reload --host 127.0.0.1 --port 8000")
            return

        try:
            env_client = PharmaSimClient(base_url=SERVER_URL, verbose=False)
            print("✓ Local server started successfully.")
        except ConnectionError as reconnect_error:
            print(f"\n✗ Connection failed after auto-start: {reconnect_error}")
            print("   Start the server manually with:")
            print("   uvicorn server.app:app --reload --host 127.0.0.1 --port 8000")
            if server_process.poll() is None:
                server_process.terminate()
            return
    
    scores = []

    try:
        for i in range(num_episodes):
            print(f"\n--- Episode {i+1}/{num_episodes} ---")
            reward = run_single_episode(env_client)
            scores.append(reward)

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Episodes completed: {len(scores)}")
        print(f"Average score: {sum(scores)/len(scores):.3f}")
        print(f"Min score: {min(scores):.3f}")
        print(f"Max score: {max(scores):.3f}")
        print(f"Success rate (score >= 0.7): {sum(1 for s in scores if s >= 0.7)/len(scores)*100:.1f}%")
        print("=" * 60)

    finally:
        if env_client is not None:
            env_client.close()
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()


if __name__ == "__main__":
    # Check if environment variables are set
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("HF_TOKEN"):
        print("\n⚠️  WARNING: No API key found!")
        print("Set OPENAI_API_KEY or HF_TOKEN environment variable")
        print("Example: export OPENAI_API_KEY='sk-...'")
        print("\nRunning with heuristic fallback instead...\n")
    
    # Run baseline
    num_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_baseline(num_episodes=num_eps)

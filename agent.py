"""
agent.py - Fully Autonomous Self-Healing HPC Agent
====================================================
Runs as a background process (nohup). Does everything itself:
  1. Detects environment (modules, conda envs, packages)
  2. Creates/fixes conda env automatically
  3. Fixes train_cnn.py and submit_job.sh if errors found
  4. Submits SLURM job
  5. Monitors job, reads .err log, patches errors, resubmits
  6. Parses results, picks best epoch/trial
  7. Calls Groq AI and APPLIES optimizations to train_cnn.py
  8. Re-runs with optimized config automatically

Usage:
  nohup python agent.py > agent_run.log 2>&1 &
  tail -f agent_run.log

Only stops if:
  - Max retries exceeded (default 5)
  - Job succeeds and optimization is applied
"""

import os, sys, re, json, time, subprocess, logging, requests, shutil
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
RESULTS_DIR    = "results"
LOGS_DIR       = "logs"
DATA_DIR       = "data"
MAX_RETRIES    = 5
POLL_INTERVAL  = 30   # seconds between squeue checks
CONDA_ENV      = "cnn_hpo"
CONDA_MODULE   = "miniconda3/24.11.1"
REQUIRED_PKGS  = ["torch", "torchvision", "optuna", "numpy", "requests"]

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AGENT] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_run.log"),
    ]
)
log = logging.getLogger(__name__)

def banner(msg):
    log.info("=" * 60)
    log.info(f"  {msg}")
    log.info("=" * 60)

# ─── Shell helpers ────────────────────────────────────────────────────────────
def run(cmd, check=False, capture=True, env=None):
    log.info(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=capture,
                       text=True, env=env or os.environ.copy())
    if r.stdout.strip(): log.info(r.stdout.strip())
    if r.stderr.strip(): log.info(r.stderr.strip())
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{r.stderr}")
    return r

def run_in_env(cmd):
    """Run command inside the conda env."""
    full = f"source activate {CONDA_ENV} 2>/dev/null || conda activate {CONDA_ENV} 2>/dev/null; {cmd}"
    return run(f'bash -c "{full}"')

# ─── Step 1: Environment Setup ────────────────────────────────────────────────
def setup_environment():
    banner("STEP 1: Environment Setup")

    # Load miniconda module
    available = run("module avail 2>&1").stdout + run("module avail 2>&1").stderr
    conda_mod = None
    for mod in ["miniconda3/25.9.1", "miniconda3/24.11.1", "anaconda3/2024.06"]:
        if mod.split("/")[0] in available or mod in available:
            conda_mod = mod
            break
    if not conda_mod:
        conda_mod = CONDA_MODULE
    log.info(f"Loading module: {conda_mod}")
    run(f"module load {conda_mod}")

    # Source conda into current shell
    conda_sh = run("conda info --base 2>/dev/null || echo /shared/EL9/explorer/miniconda3/24.11.1/miniconda3").stdout.strip()
    conda_sh = conda_sh.rstrip("/") + "/etc/profile.d/conda.sh"
    if not os.path.exists(conda_sh):
        # try to find it
        r = run("find /shared -name 'conda.sh' 2>/dev/null | head -1")
        conda_sh = r.stdout.strip() or conda_sh

    # Check if env exists
    envs = run("conda env list 2>/dev/null").stdout
    env_exists = os.path.isdir(os.path.expanduser(f"~/.conda/envs/{CONDA_ENV}"))
    if not env_exists:
        log.info(f"Creating conda env: {CONDA_ENV}")
        run(f"bash -c 'source {conda_sh} && conda create -n {CONDA_ENV} python=3.10 -y'", check=True)
    else:
        log.info(f"Conda env '{CONDA_ENV}' already exists")

    # Install missing packages
    for pkg in REQUIRED_PKGS:
        r = run(f"bash -c 'source {conda_sh} && conda activate {CONDA_ENV} && python -c \"import {pkg.split('[')[0]}\"'")
        if r.returncode != 0:
            log.info(f"Installing missing package: {pkg}")
            run(f"bash -c 'source {conda_sh} && conda activate {CONDA_ENV} && pip install {pkg}'")

    # Write a helper activate script for SLURM
    with open("env_activate.sh", "w") as f:
        f.write(f"source {conda_sh}\nconda activate {CONDA_ENV}\n")

    log.info("Environment setup complete")
    return conda_sh

# ─── Step 2: Fix submit_job.sh ────────────────────────────────────────────────
def fix_slurm_script(conda_sh):
    banner("STEP 2: Fixing SLURM Script")

    # Detect available cuda module
    avail = run("module avail 2>&1").stderr + run("module avail 2>&1").stdout
    cuda_mod = "cuda/12.3.0"
    for line in avail.split():
        if line.startswith("cuda/"):
            cuda_mod = line.strip()
            break

    script = f"""#!/bin/bash
#SBATCH --job-name=cnn_hpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output={LOGS_DIR}/cnn_hpo_%j.out
#SBATCH --error={LOGS_DIR}/cnn_hpo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=murugan.g@northeastern.edu

echo "============================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Start time    : $(date)"
echo "Working dir   : $SLURM_SUBMIT_DIR"
echo "============================================================"

cd $SLURM_SUBMIT_DIR
mkdir -p {LOGS_DIR} {RESULTS_DIR} {DATA_DIR}

module load {cuda_mod} 2>/dev/null || true
source {conda_sh}
conda activate {CONDA_ENV}

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"

echo ""
echo ">>> Step 1: Generating dataset..."
python generate_dataset.py

echo ""
echo ">>> Step 2: Training CNN with Optuna HPO..."
python train_cnn.py \\
    --data_path  {DATA_DIR}/dataset.npz \\
    --output_dir {RESULTS_DIR}/ \\
    --n_trials   20 \\
    --max_epochs 30

echo ""
echo ">>> Step 3: Summary"
python -c "
import json, sys
try:
    with open('{RESULTS_DIR}/summary.json') as f:
        s = json.load(f)
    print(f'Best Trial   : {{s[chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(116)+chr(114)+chr(105)+chr(97)+chr(108)]}}')
    print(f'Best Epoch   : {{s[chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(101)+chr(112)+chr(111)+chr(99)+chr(104)]}}')
    print(f'Best Val Acc : {{s[chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(118)+chr(97)+chr(108)+chr(95)+chr(97)+chr(99)+chr(99)]:.5f}}')
except Exception as e:
    print(f'Summary error: {{e}}', file=sys.stderr)
    sys.exit(1)
"

echo "============================================================"
echo "Job complete: $(date)"
echo "============================================================"
"""
    # Simplify the summary python (avoid chr obfuscation)
    script = script.replace(
        """    print(f'Best Trial   : {{s[chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(116)+chr(114)+chr(105)+chr(97)+chr(108)]}}')
    print(f'Best Epoch   : {{s[chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(101)+chr(112)+chr(111)+chr(99)+chr(104)]}}')
    print(f'Best Val Acc : {{s[chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(118)+chr(97)+chr(108)+chr(95)+chr(97)+chr(99)+chr(99)]:.5f}}')""",
        """    print(f"Best Trial   : {s['best_trial']}")
    print(f"Best Epoch   : {s['best_epoch']}")
    print(f"Best Val Acc : {s['best_val_acc']:.5f}")"""
    )

    with open("submit_job.sh", "w") as f:
        f.write(script)
    run("chmod +x submit_job.sh")
    log.info("submit_job.sh written and fixed")

# ─── Step 3: Fix train_cnn.py imports ────────────────────────────────────────
def fix_train_script():
    banner("STEP 3: Validating train_cnn.py")
    with open("train_cnn.py") as f:
        content = f.read()

    changed = False

    # Fix JournalFileBackend import (removed in optuna 4.x)
    if "JournalFileBackend" in content or "JournalStorage" in content:
        content = re.sub(r'from optuna\.storages import JournalStorage.*?\n', '', content)
        content = re.sub(r'try:.*?JournalFileBackend.*?except.*?\n', '', content, flags=re.DOTALL)
        content = re.sub(r'from optuna\.storages\._journal.*?\n', '', content)
        changed = True
        log.info("Fixed: removed JournalFileBackend import")

    # Fix storage line
    if "storage = JournalStorage" in content:
        content = re.sub(
            r'storage\s*=\s*JournalStorage\(.*?\)',
            'storage = f"sqlite:///{os.path.join(args.output_dir, \'optuna.db\')}"',
            content, flags=re.DOTALL
        )
        changed = True
        log.info("Fixed: replaced JournalStorage with SQLite")

    if "storage = None" in content:
        content = content.replace(
            "storage = None",
            'storage = f"sqlite:///{os.path.join(args.output_dir, \'optuna.db\')}"'
        )
        changed = True
        log.info("Fixed: replaced None storage with SQLite")

    if changed:
        with open("train_cnn.py", "w") as f:
            f.write(content)
        log.info("train_cnn.py patched successfully")
    else:
        log.info("train_cnn.py looks good, no changes needed")

# ─── Step 4: Submit SLURM Job ─────────────────────────────────────────────────
def submit_job():
    banner("STEP 4: Submitting SLURM Job")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    r = run("sbatch submit_job.sh", check=True)
    match = re.search(r'Submitted batch job (\d+)', r.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from: {r.stdout}")
    job_id = match.group(1)
    log.info(f"Job submitted: {job_id}")
    return job_id

# ─── Step 5: Monitor + Auto-Heal ─────────────────────────────────────────────
def monitor_and_heal(job_id, conda_sh, retry_count=0):
    banner(f"STEP 5: Monitoring Job {job_id} (retry {retry_count}/{MAX_RETRIES})")

    while True:
        time.sleep(POLL_INTERVAL)
        r = run(f"squeue -j {job_id} -h -o '%T'")
        status = r.stdout.strip()
        log.info(f"Job {job_id} status: {status or 'COMPLETED/UNKNOWN'}")

        if status in ("RUNNING", "PENDING", "CONFIGURING"):
            # Peek at err log for early errors
            err_file = f"{LOGS_DIR}/cnn_hpo_{job_id}.err"
            if os.path.exists(err_file):
                err = open(err_file).read()
                if "Error" in err or "error" in err or "Traceback" in err:
                    log.info("Detected error in .err log while job running — will heal after completion")
            continue

        # Job finished (or unknown = finished)
        break

    # Check outcome
    err_file = f"{LOGS_DIR}/cnn_hpo_{job_id}.err"
    out_file = f"{LOGS_DIR}/cnn_hpo_{job_id}.out"
    summary_exists = os.path.exists(f"{RESULTS_DIR}/summary.json")

    if summary_exists:
        log.info("Job completed successfully — summary.json found")
        return True

    # Job failed — diagnose and heal
    log.info("Job did not produce summary.json — diagnosing errors...")
    if not os.path.exists(err_file):
        log.info("No .err file found yet, waiting 10s...")
        time.sleep(10)

    err_content = open(err_file).read() if os.path.exists(err_file) else ""
    out_content = open(out_file).read() if os.path.exists(out_file) else ""

    if retry_count >= MAX_RETRIES:
        log.error(f"Max retries ({MAX_RETRIES}) exceeded. Giving up.")
        return False

    healed = auto_heal(err_content, out_content, conda_sh)
    if healed:
        log.info("Auto-heal applied — resubmitting job...")
        new_job_id = submit_job()
        return monitor_and_heal(new_job_id, conda_sh, retry_count + 1)
    else:
        log.error("Could not auto-heal. Manual intervention needed.")
        log.error(f"Error log:\n{err_content}")
        return False

# ─── Auto-Heal Engine ─────────────────────────────────────────────────────────
def auto_heal(err, out, conda_sh):
    """Reads error output and applies fixes automatically. Returns True if healed."""
    log.info("Running auto-heal engine...")
    healed = False

    # Fix 1: conda not initialized
    if "conda init" in err or "EnvironmentLocationNotFound" in err or "CondaError" in err:
        log.info("HEAL: Fixing conda activation in submit_job.sh")
        fix_slurm_script(conda_sh)
        healed = True

    # Fix 2: module not found
    if "Unable to locate a modulefile" in err or "Unable to locate" in err:
        log.info("HEAL: Fixing module names in submit_job.sh")
        fix_slurm_script(conda_sh)
        healed = True

    # Fix 3: ImportError / ModuleNotFoundError
    if "ModuleNotFoundError" in err or "ImportError" in err:
        # Find which module is missing
        missing = re.findall(r"No module named '([^']+)'", err)
        missing += re.findall(r"cannot import name '([^']+)'", err)
        for mod in missing:
            if "JournalFileBackend" in mod or "optuna" in mod:
                log.info(f"HEAL: Fixing optuna import issue")
                fix_train_script()
                healed = True
            else:
                log.info(f"HEAL: Installing missing package: {mod}")
                run(f"bash -c 'source {conda_sh} && conda activate {CONDA_ENV} && pip install {mod}'")
                healed = True

    # Fix 4: CUDA driver mismatch — fall back to CPU gracefully
    if "CUDA initialization" in err or "cudaErrorInsufficientDriver" in err:
        log.info("HEAL: CUDA driver mismatch — patching train_cnn.py to force CPU fallback")
        with open("train_cnn.py") as f:
            content = f.read()
        content = content.replace(
            'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
            'device = torch.device("cpu")  # forced: CUDA driver mismatch on this node'
        )
        with open("train_cnn.py", "w") as f:
            f.write(content)
        healed = True

    # Fix 5: FileNotFoundError for data or results
    if "FileNotFoundError" in err:
        missing_files = re.findall(r"No such file or directory: '([^']+)'", err)
        for mf in missing_files:
            d = os.path.dirname(mf)
            if d:
                os.makedirs(d, exist_ok=True)
                log.info(f"HEAL: Created missing directory: {d}")
        healed = True

    # Fix 6: OOM / memory error
    if "OutOfMemoryError" in err or "CUDA out of memory" in err:
        log.info("HEAL: OOM detected — reducing batch size in train_cnn.py")
        with open("train_cnn.py") as f:
            content = f.read()
        content = re.sub(
            r'suggest_categorical\("batch_size", \[.*?\]\)',
            'suggest_categorical("batch_size", [16, 32])',
            content
        )
        with open("train_cnn.py", "w") as f:
            f.write(content)
        healed = True

    # Fix 7: Timeout — reduce trials/epochs
    if "TIMEOUT" in out or "DUE TO TIME LIMIT" in err:
        log.info("HEAL: Job timed out — reducing n_trials and max_epochs in submit_job.sh")
        with open("submit_job.sh") as f:
            s = f.read()
        s = re.sub(r'--n_trials\s+\d+', '--n_trials 10', s)
        s = re.sub(r'--max_epochs\s+\d+', '--max_epochs 15', s)
        with open("submit_job.sh", "w") as f:
            f.write(s)
        healed = True

    if not healed:
        log.info("HEAL: No known fix found for this error pattern")

    return healed

# ─── Step 6: Parse Results ────────────────────────────────────────────────────
def parse_results():
    banner("STEP 6: Parsing Results")
    with open(f"{RESULTS_DIR}/summary.json") as f:
        summary = json.load(f)

    log.info(f"Best Trial   : {summary['best_trial']}")
    log.info(f"Best Epoch   : {summary['best_epoch']}")
    log.info(f"Best Val Acc : {summary['best_val_acc']:.5f}")
    log.info(f"Best Params  : {summary['best_params']}")
    log.info("Top 5 Trials:")
    for i, t in enumerate(summary["all_trials_ranked"][:5], 1):
        log.info(f"  #{i} Trial {t['trial']} val_acc={t['best_val_acc']:.4f} epoch={t['best_epoch']}")
    return summary

# ─── Step 7: Groq AI → Auto-Apply Optimizations ──────────────────────────────
def optimize_and_apply(summary):
    banner("STEP 7: AI Optimization + Auto-Apply")

    if not GROQ_API_KEY:
        log.warning("GROQ_API_KEY not set — skipping AI optimization")
        return

    # Load epoch logs
    epoch_snippets = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith("_epochs.json") and len(epoch_snippets) < 3:
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                epoch_snippets.append(json.load(f))

    prompt = f"""You are an expert ML engineer. Analyze these CNN training results and return ONLY a JSON object with concrete parameter changes to apply. No explanations, no markdown, just valid JSON.

Results:
- Best val_acc: {summary['best_val_acc']:.5f}
- Best params: {json.dumps(summary['best_params'])}
- Top 5 trials: {json.dumps(summary['all_trials_ranked'][:5])}
- Sample epoch logs: {json.dumps(epoch_snippets, indent=2)[:2000]}

Return JSON in exactly this format:
{{
  "n_trials": <integer, suggested number of trials>,
  "max_epochs": <integer, suggested max epochs>,
  "lr_min": <float, min learning rate>,
  "lr_max": <float, max learning rate>,
  "filters": [<int>, <int>, <int>],
  "batch_sizes": [<int>, <int>],
  "dropout_min": <float>,
  "dropout_max": <float>,
  "reason": "<one sentence why>"
}}"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an ML expert. Return only valid JSON, no markdown, no explanation."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 500,
        "temperature": 0.1,
    }

    log.info("Calling Groq AI for optimization parameters...")
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        raw = re.sub(r'```json|```', '', raw).strip()
        params = json.loads(raw)
        log.info(f"AI suggested params: {params}")
    except Exception as e:
        log.error(f"Groq API error: {e}")
        return

    # Auto-apply to train_cnn.py
    log.info("Auto-applying optimizations to train_cnn.py...")
    with open("train_cnn.py") as f:
        content = f.read()

    if "lr_min" in params and "lr_max" in params:
        content = re.sub(
            r'suggest_float\("lr",\s*[\d.e-]+,\s*[\d.e-]+',
            f'suggest_float("lr", {params["lr_min"]}, {params["lr_max"]}',
            content
        )
        log.info(f"  Applied lr range: {params['lr_min']} - {params['lr_max']}")

    if "filters" in params and len(params["filters"]) >= 2:
        f1 = str(params["filters"])
        content = re.sub(
            r'suggest_categorical\("n_filters1",\s*\[.*?\]\)',
            f'suggest_categorical("n_filters1", {f1})',
            content
        )
        content = re.sub(
            r'suggest_categorical\("n_filters2",\s*\[.*?\]\)',
            f'suggest_categorical("n_filters2", {f1})',
            content
        )
        log.info(f"  Applied filter sizes: {f1}")

    if "batch_sizes" in params:
        content = re.sub(
            r'suggest_categorical\("batch_size",\s*\[.*?\]\)',
            f'suggest_categorical("batch_size", {params["batch_sizes"]})',
            content
        )
        log.info(f"  Applied batch sizes: {params['batch_sizes']}")

    if "dropout_min" in params and "dropout_max" in params:
        content = re.sub(
            r'suggest_float\("dropout",\s*[\d.]+,\s*[\d.]+\)',
            f'suggest_float("dropout", {params["dropout_min"]}, {params["dropout_max"]})',
            content
        )
        log.info(f"  Applied dropout range: {params['dropout_min']} - {params['dropout_max']}")

    with open("train_cnn.py", "w") as f:
        f.write(content)
    log.info("train_cnn.py updated with AI-optimized parameters")

    # Auto-apply n_trials and max_epochs to submit_job.sh
    if "n_trials" in params or "max_epochs" in params:
        with open("submit_job.sh") as f:
            slurm = f.read()
        if "n_trials" in params:
            slurm = re.sub(r'--n_trials\s+\d+', f'--n_trials {params["n_trials"]}', slurm)
        if "max_epochs" in params:
            slurm = re.sub(r'--max_epochs\s+\d+', f'--max_epochs {params["max_epochs"]}', slurm)
        with open("submit_job.sh", "w") as f:
            f.write(slurm)
        log.info(f"  submit_job.sh: n_trials={params.get('n_trials')}, max_epochs={params.get('max_epochs')}")

    # Save what was applied
    with open(f"{RESULTS_DIR}/applied_optimizations.json", "w") as f:
        json.dump(params, f, indent=2)

    log.info(f"Reason: {params.get('reason', 'N/A')}")
    return params

# ─── Step 8: Run Optimized Job ────────────────────────────────────────────────
def run_optimized_job(conda_sh):
    banner("STEP 8: Running Optimized Job")
    # Back up old results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.exists(RESULTS_DIR):
        shutil.copytree(RESULTS_DIR, f"{RESULTS_DIR}_backup_{ts}")
        log.info(f"Old results backed up to results_backup_{ts}/")
    # Clean results for fresh run
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json") or f.endswith(".pt") or f.endswith(".db"):
            os.remove(os.path.join(RESULTS_DIR, f))

    job_id = submit_job()
    success = monitor_and_heal(job_id, conda_sh)
    if success:
        summary = parse_results()
        log.info("Optimized run complete!")
        log.info(f"Final best val_acc: {summary['best_val_acc']:.5f}")
    return success

# ─── Main Orchestrator ────────────────────────────────────────────────────────
def main():
    banner("AUTONOMOUS HPC CNN AGENT STARTING")
    log.info(f"Working directory: {os.getcwd()}")
    log.info(f"Time: {datetime.now()}")

    # Step 1: Environment
    conda_sh = setup_environment()

    # Step 2 & 3: Fix scripts
    fix_slurm_script(conda_sh)
    fix_train_script()

    # Step 4: Submit
    job_id = submit_job()

    # Step 5: Monitor + heal
    success = monitor_and_heal(job_id, conda_sh)
    if not success:
        log.error("Pipeline failed after max retries. Check agent_run.log.")
        sys.exit(1)

    # Step 6: Parse results
    summary = parse_results()

    # Step 7: AI optimize + auto-apply
    optimize_and_apply(summary)

    # Step 8: Re-run with optimized config
    run_optimized_job(conda_sh)

    banner("PIPELINE COMPLETE")
    log.info("All results in results/ and results_backup_*/")
    log.info("Check agent_run.log for full audit trail")

if __name__ == "__main__":
    main()

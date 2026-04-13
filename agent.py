"""
agent.py
Claude-powered AI Agent that:
  1. Generates the dataset
  2. Submits SLURM job to Explorer
  3. Monitors job status
  4. Reads results, picks best epoch/trial
  5. Suggests optimizations via Claude API

Usage:
  python agent.py [--submit] [--monitor JOB_ID] [--optimize]
"""
import argparse
import json
import os
import subprocess
import time
import sys
import requests

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RESULTS_DIR = "results"

# ─── SLURM Helpers ───────────────────────────────────────────────────────────

def submit_job():
    print(">>> Submitting SLURM job...")
    result = subprocess.run(
        ["sbatch", "submit_job.sh"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] sbatch failed:\n{result.stderr}")
        sys.exit(1)
    # "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    print(f">>> Job submitted: {job_id}")
    return job_id

def check_job_status(job_id):
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True, text=True
    )
    status = result.stdout.strip()
    return status if status else "COMPLETED/UNKNOWN"

def monitor_job(job_id, poll_interval=60):
    print(f">>> Monitoring job {job_id} (polling every {poll_interval}s)...")
    while True:
        status = check_job_status(job_id)
        print(f"  [{time.strftime('%H:%M:%S')}] Job {job_id} status: {status}")
        if status in ("COMPLETED/UNKNOWN", "FAILED", "CANCELLED", "TIMEOUT"):
            break
        time.sleep(poll_interval)
    print(f">>> Job {job_id} finished with status: {status}")
    return status

# ─── Results Parser ───────────────────────────────────────────────────────────

def parse_results():
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print("[WARN] No summary.json found. Training may not be complete.")
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    print("\n" + "=" * 60)
    print("BEST RESULTS")
    print("=" * 60)
    print(f"  Best Trial   : {summary['best_trial']}")
    print(f"  Best Epoch   : {summary['best_epoch']}")
    print(f"  Best Val Acc : {summary['best_val_acc']:.5f}")
    print(f"  Best Params  : {summary['best_params']}")
    print("\nTop 5 Trials:")
    for i, t in enumerate(summary["all_trials_ranked"][:5], 1):
        print(f"  #{i}  Trial {t['trial']}  val_acc={t['best_val_acc']:.4f}  epoch={t['best_epoch']}")
    print("=" * 60)
    return summary

# ─── Claude Optimization Agent ────────────────────────────────────────────────

def ask_claude_for_optimizations(summary):
    if not ANTHROPIC_API_KEY:
        print("[WARN] ANTHROPIC_API_KEY not set. Skipping Claude optimization.")
        return

    # Load a few epoch logs for context
    epoch_snippets = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith("_epochs.json") and len(epoch_snippets) < 3:
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                epoch_snippets.append(json.load(f))

    prompt = f"""
You are an expert ML engineer reviewing CNN training results on an HPC cluster.

## Experiment Summary
- Task: Image classification (3 classes, 32x32 RGB synthetic dataset)
- Model: SmallCNN (2-3 conv layers + FC)
- HPO: Optuna ({len(summary['all_trials_ranked'])} trials, up to 30 epochs each)
- Best Val Acc: {summary['best_val_acc']:.5f}
- Best Trial: {summary['best_trial']}, Best Epoch: {summary['best_epoch']}
- Best Params: {json.dumps(summary['best_params'], indent=2)}

## Sample Epoch Logs (first 3 trials)
{json.dumps(epoch_snippets, indent=2)[:3000]}

## Top 5 Trials Ranked
{json.dumps(summary['all_trials_ranked'][:5], indent=2)}

Based on the above:
1. Diagnose any overfitting, underfitting, or training instability.
2. Suggest 3-5 concrete hyperparameter changes (with specific values) to improve val_acc.
3. Suggest any architectural improvements to the CNN.
4. Recommend whether to run more Optuna trials, increase epochs, or try a different approach.
5. Suggest any SLURM/HPC optimization (resource allocation, parallelism).

Be concise and specific. Format as numbered recommendations.
"""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }

    print("\n>>> Asking Claude for optimization recommendations...")
    resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
    if resp.status_code != 200:
        print(f"[ERROR] Claude API error: {resp.text}")
        return

    text = resp.json()["content"][0]["text"]
    print("\n" + "=" * 60)
    print("CLAUDE'S OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    print(text)
    print("=" * 60)

    # Save to file
    opt_path = os.path.join(RESULTS_DIR, "optimization_recommendations.txt")
    with open(opt_path, "w") as f:
        f.write(text)
    print(f"\n>>> Saved recommendations to {opt_path}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CNN HPC AI Agent")
    parser.add_argument("--submit",   action="store_true", help="Generate data and submit SLURM job")
    parser.add_argument("--monitor",  type=str, default=None, metavar="JOB_ID", help="Monitor a running job")
    parser.add_argument("--optimize", action="store_true", help="Parse results and ask Claude for optimizations")
    parser.add_argument("--all",      action="store_true", help="Run full pipeline: submit → monitor → optimize")
    args = parser.parse_args()

    if args.all or args.submit:
        # Generate dataset first
        print(">>> Generating dataset...")
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)
        job_id = submit_job()

        if args.all:
            status = monitor_job(job_id)
            if status not in ("FAILED", "CANCELLED", "TIMEOUT"):
                summary = parse_results()
                if summary:
                    ask_claude_for_optimizations(summary)
        else:
            print(f"\nTo monitor: python agent.py --monitor {job_id}")
            print(f"To optimize after: python agent.py --optimize")

    elif args.monitor:
        monitor_job(args.monitor)

    elif args.optimize:
        summary = parse_results()
        if summary:
            ask_claude_for_optimizations(summary)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

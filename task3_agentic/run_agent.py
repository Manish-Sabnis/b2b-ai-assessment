"""
Task 3 - Main Runner
Demonstrates the full agentic pipeline with a simulated batch.
Includes one example of QC Agent triggering re-inference (required deliverable).

Usage:
    python run_agent.py                        # demo with mock images
    python run_agent.py --image_dir /path/to/images  # real images
    python run_agent.py --simulate_drift       # simulate drift scenario
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent))

from graph import run_pipeline
from memory.memory import LongTermMemory


def make_demo_images(n_crop: int = 4, n_traffic: int = 3,
                     n_low_quality: int = 2) -> list:
    paths = []
    for i in range(n_crop):
        paths.append(f"demo_images/crop_field_{i+1:03d}.jpg")
    for i in range(n_traffic):
        paths.append(f"demo_images/traffic_urban_{i+1:03d}.jpg")
    for i in range(n_low_quality):
       
        paths.append(f"demo_images/foggy_scene_{i+1:03d}.jpg")
    return paths


def simulate_drift_history(lt_memory: LongTermMemory, n_batches: int = 5):
    """
    Pre-populate long-term memory with declining confidence scores
    to demonstrate drift detection triggering.
    """
    print("\n[Demo] Simulating batch history for drift detection...")
    import random
    random.seed(42)
    for i in range(n_batches):
        conf = 0.72 - (i * 0.06) + random.gauss(0, 0.02)
        lt_memory.log_batch({
            "batch_id":        f"historical_batch_{i+1:03d}",
            "total_processed": 10,
            "crop_images":     5,
            "traffic_images":  5,
            "mean_confidence": round(max(0.1, conf), 4),
            "flagged":         i,
            "reprocessed":     max(0, i - 1),
        })
    print(f"  Injected {n_batches} historical batches with declining confidence.")


def print_final_summary(final_state: dict):
    """Print a clean summary of the pipeline run."""
    summary = final_state.get("batch_summary", {})
    drift   = final_state.get("drift_status", {})
    qc      = final_state.get("qc_results", [])

    accepted    = sum(1 for r in qc if r.get("decision") == "accept")
    reinfered   = sum(1 for r in qc if r.get("decision") == "reinfer")
    escalated   = sum(1 for r in qc if r.get("decision") == "escalate")

    print(f"\n{'='*60}")
    print("PIPELINE RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Batch ID       : {summary.get('batch_id', 'N/A')}")
    print(f"Total processed: {summary.get('total_processed', 0)}")
    print(f"  Crop images  : {summary.get('crop_images', 0)}")
    print(f"  Traffic imgs : {summary.get('traffic_images', 0)}")
    print(f"Mean confidence: {summary.get('mean_confidence', 0):.4f}")
    print(f"\nQC Results:")
    print(f"  Accepted     : {accepted}")
    print(f"  Re-inferred  : {reinfered}  ← QC Agent triggered re-inference")
    print(f"  Escalated    : {escalated}")
    print(f"\nDrift Status:")
    print(f"  Detected     : {drift.get('drift_detected', False)}")
    print(f"  Score        : {drift.get('drift_score', 'N/A')}")
    print(f"  Retrain flag : {drift.get('retrain_flag', False)}")
    print(f"\nOutputs saved to: outputs/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3 - Agentic Pipeline Runner")
    parser.add_argument("--image_dir",      default=None,
                        help="Directory of real images (uses mock if not set)")
    parser.add_argument("--batch_id",       default=None,
                        help="Custom batch ID")
    parser.add_argument("--simulate_drift", action="store_true",
                        help="Pre-populate history to trigger drift detection")
    parser.add_argument("--n_images",       type=int, default=9,
                        help="Number of demo images to process")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("memory",  exist_ok=True)
    os.makedirs("logs",    exist_ok=True)


    if args.simulate_drift:
        lt = LongTermMemory()
        simulate_drift_history(lt, n_batches=5)

    if args.image_dir and Path(args.image_dir).exists():
        exts   = [".jpg", ".jpeg", ".png"]
        images = [str(p) for p in Path(args.image_dir).iterdir()
                  if p.suffix.lower() in exts][:args.n_images]
        print(f"Using {len(images)} real images from {args.image_dir}")
    else:
        images = make_demo_images(n_crop=4, n_traffic=3, n_low_quality=2)
        print(f"Using {len(images)} mock image paths for demo")

    batch_id    = args.batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    final_state = run_pipeline(images, batch_id=batch_id)
    print_final_summary(final_state)


    state_path = f"outputs/{batch_id}_full_state.json"
    with open(state_path, "w") as f:
        safe_state = {k: v for k, v in final_state.items()
                      if isinstance(v, (str, int, float, list, dict, bool, type(None)))}
        json.dump(safe_state, f, indent=2)
    print(f"Full state saved → {state_path}")
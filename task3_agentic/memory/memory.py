"""
Task 3 - Memory Module
Short-term: within-batch state (current run context)
Long-term:  cross-batch metrics store (JSON file on disk)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


MEMORY_PATH = Path(__file__).parent.parent / "memory" / "long_term_store.json"


class ShortTermMemory:
    """
    Holds state within a single batch run.
    Cleared between batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.batch_id         = None
        self.processed        = []      
        self.flagged          = []      
        self.reprocessed      = []      
        self.routing_decisions= {}      
        self.start_time       = datetime.now().isoformat()

    def log_result(self, result: Dict):
        self.processed.append(result)

    def flag_for_reinference(self, image_path: str, reason: str):
        self.flagged.append({"image": image_path, "reason": reason,
                              "timestamp": datetime.now().isoformat()})

    def log_reprocessed(self, result: Dict):
        self.reprocessed.append(result)

    def log_routing(self, image_path: str, domain: str):
        self.routing_decisions[image_path] = domain

    def summary(self) -> Dict:
        total = len(self.processed)
        if total == 0:
            return {"total": 0}

        crop_results    = [r for r in self.processed if r.get("task") == "crop_detection"]
        traffic_results = [r for r in self.processed if r.get("task") == "traffic_detection"]

        return {
            "batch_id":        self.batch_id,
            "total_processed": total,
            "crop_images":     len(crop_results),
            "traffic_images":  len(traffic_results),
            "flagged":         len(self.flagged),
            "reprocessed":     len(self.reprocessed),
            "mean_confidence": round(
                sum(r.get("mean_confidence", 0) for r in self.processed) / total, 4
            ),
            "start_time": self.start_time,
            "end_time":   datetime.now().isoformat(),
        }


class LongTermMemory:
    """
    Persists batch metrics across runs.
    Used by QC agent for drift detection.
    """

    def __init__(self, path: str = str(MEMORY_PATH)):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store = self._load()

    def _load(self) -> Dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"batches": [], "drift_events": [], "created": datetime.now().isoformat()}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._store, f, indent=2)

    def log_batch(self, batch_summary: Dict):
        """Save batch metrics after each run."""
        batch_summary["logged_at"] = datetime.now().isoformat()
        self._store["batches"].append(batch_summary)
        self._save()

    def log_drift_event(self, event: Dict):
        """Record a model drift event."""
        event["timestamp"] = datetime.now().isoformat()
        self._store["drift_events"].append(event)
        self._save()

    def get_recent_batches(self, n: int = 5) -> List[Dict]:
        return self._store["batches"][-n:]

    def get_drift_events(self) -> List[Dict]:
        return self._store["drift_events"]

    def compute_drift_score(self, metric: str = "mean_confidence",
                             window: int = 5) -> Optional[float]:
        """
        Compare latest batch metric against rolling average of previous batches.
        Returns drift score (negative = degradation).
        """
        batches = self.get_recent_batches(window + 1)
        if len(batches) < 2:
            return None
        latest   = batches[-1].get(metric, 0)
        baseline = sum(b.get(metric, 0) for b in batches[:-1]) / (len(batches) - 1)
        return round(latest - baseline, 4)

    def should_retrain(self,
                       confidence_threshold: float = 0.45,
                       drift_threshold: float = -0.08) -> Dict:
        """
        Check if retraining should be triggered based on:
        1. Mean confidence below threshold
        2. Drift score below threshold
        """
        recent = self.get_recent_batches(3)
        if not recent:
            return {"retrain": False, "reason": "insufficient history"}

        latest_conf  = recent[-1].get("mean_confidence", 1.0)
        drift_score  = self.compute_drift_score()

        if latest_conf < confidence_threshold:
            return {"retrain": True,
                    "reason": f"confidence {latest_conf:.3f} < threshold {confidence_threshold}",
                    "metric": "mean_confidence", "value": latest_conf}
        if drift_score is not None and drift_score < drift_threshold:
            return {"retrain": True,
                    "reason": f"drift score {drift_score:.3f} < threshold {drift_threshold}",
                    "metric": "drift_score", "value": drift_score}

        return {"retrain": False, "reason": "metrics within acceptable range",
                "latest_confidence": latest_conf,
                "drift_score": drift_score}

    def full_report(self) -> Dict:
        batches = self._store["batches"]
        if not batches:
            return {"status": "no data"}
        return {
            "total_batches":   len(batches),
            "drift_events":    len(self._store["drift_events"]),
            "latest_batch":    batches[-1] if batches else None,
            "drift_score":     self.compute_drift_score(),
            "retrain_check":   self.should_retrain(),
        }
"""
Task 3 - Agent Definitions
Three agents built with LangChain + Gemini:
  1. OrchestratorAgent  — routes images, schedules inference
  2. QualityControlAgent — validates confidence, triggers re-inference
  3. ReportingAgent      — generates NL summaries, anomaly alerts
"""

import os
import json
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def get_llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=True,
    )




class OrchestratorAgent:
    """
    Routes incoming images to the correct CV pipeline (crop vs traffic).
    Schedules inference jobs and manages batch state.
    """

    SYSTEM_PROMPT = """You are an Orchestrator Agent for an aerial imagery AI system.
Your job is to:
1. Analyse image metadata and decide whether it should go to the CROP detection pipeline or TRAFFIC detection pipeline
2. Explain your routing decision clearly
3. Flag any preprocessing concerns (resolution, missing metadata, unusual conditions)

Respond in JSON format:
{
  "domain": "crop" or "traffic",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "preprocessing_flags": ["list of concerns or empty list"],
  "priority": "high" or "normal"
}"""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)

    def route(self, image_path: str, metadata: Dict = None) -> Dict:
        """Decide which pipeline to route an image to."""
        from tools.cv_tools import classify_image_domain
        domain_hint = classify_image_domain(image_path)
        meta_str = json.dumps(metadata or {}, indent=2)

        prompt = f"""Route this aerial image to the correct CV pipeline.

Image path: {image_path}
Keyword hint: {domain_hint}
Metadata: {meta_str}

Decide: should this go to CROP detection or TRAFFIC detection?"""

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            text = response.content.strip()

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            decision = json.loads(text)
        except Exception as e:
            decision = {
                "domain":               domain_hint,
                "confidence":           0.75,
                "reasoning":            f"Keyword-based routing (LLM error: {e})",
                "preprocessing_flags":  [],
                "priority":             "normal",
            }

        decision["image_path"] = image_path
        decision["routed_at"]  = datetime.now().isoformat()
        return decision

    def schedule_batch(self, image_paths: List[str]) -> List[Dict]:
        """Route a batch of images."""
        print(f"\n[Orchestrator] Scheduling batch of {len(image_paths)} images...")
        decisions = []
        for path in image_paths:
            decision = self.route(path)
            decisions.append(decision)
            domain = decision["domain"]
            conf   = decision["confidence"]
            print(f"  → {path.split('/')[-1]:<30} domain={domain:<8} conf={conf:.2f}")
        return decisions


class QualityControlAgent:
    """
    Validates inference output quality.
    Triggers re-inference if confidence is below threshold.
    Detects model drift across batches.
    """

    SYSTEM_PROMPT = """You are a Quality Control Agent for an aerial imagery AI pipeline.
Your job is to:
1. Evaluate whether inference results meet quality thresholds
2. Identify failure modes (fog, shadow, low resolution, off-nadir angle)
3. Decide whether to accept, flag, or trigger re-inference
4. Detect performance drift patterns

Respond in JSON format:
{
  "decision": "accept" or "reinfer" or "escalate",
  "quality_score": 0.0-1.0,
  "failure_modes": ["list of detected issues or empty"],
  "reasoning": "explanation",
  "augmentation_recommendation": "suggestion for next training cycle or null"
}"""

    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
    IOU_THRESHOLD        = float(os.getenv("IOU_THRESHOLD", "0.35"))

    def __init__(self):
        self.llm = get_llm(temperature=0.05)

    def validate(self, inference_result: Dict) -> Dict:
        """Validate a single inference result."""
        mean_conf = inference_result.get("mean_confidence", 0.0)
        task      = inference_result.get("task", "unknown")

        # Fast rule-based check first
        if mean_conf >= self.CONFIDENCE_THRESHOLD:
            fast_decision = "accept"
        elif mean_conf >= self.CONFIDENCE_THRESHOLD * 0.6:
            fast_decision = "reinfer"
        else:
            fast_decision = "escalate"

        prompt = f"""Evaluate this {task} inference result:

Mean confidence: {mean_conf:.4f}
Confidence threshold: {self.CONFIDENCE_THRESHOLD}
Number of detections: {inference_result.get('n_polygons') or inference_result.get('n_vehicles', 0)}
Model: {inference_result.get('model', 'unknown')}
Image: {inference_result.get('image_path', 'unknown')}
Fast decision: {fast_decision}

Validate this result and provide your QC decision."""

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            text = response.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            qc_result = json.loads(text)
        except Exception as e:
            qc_result = {
                "decision":                  fast_decision,
                "quality_score":             mean_conf,
                "failure_modes":             [] if mean_conf > 0.4 else ["low_confidence"],
                "reasoning":                 f"Rule-based QC (LLM error: {e})",
                "augmentation_recommendation": None,
            }

        qc_result["mean_confidence"] = mean_conf
        qc_result["validated_at"]    = datetime.now().isoformat()
        qc_result["image_path"]      = inference_result.get("image_path")
        return qc_result

    def check_drift(self, long_term_memory) -> Dict:
        """Check for model performance drift across batches."""
        drift_score    = long_term_memory.compute_drift_score()
        retrain_check  = long_term_memory.should_retrain()
        recent_batches = long_term_memory.get_recent_batches(5)

        if drift_score is None:
            return {"drift_detected": False, "reason": "insufficient history"}

        if retrain_check["retrain"]:
            # Log drift event
            event = {
                "drift_score":   drift_score,
                "trigger":       retrain_check["reason"],
                "recent_batches": len(recent_batches),
            }
            long_term_memory.log_drift_event(event)

            print(f"\n[QC Agent] ⚠ DRIFT DETECTED: {retrain_check['reason']}")
            print(f"  Drift score: {drift_score:.4f}")
            print(f"  Action: Flagging for retraining")

        return {
            "drift_detected": retrain_check["retrain"],
            "drift_score":    drift_score,
            "retrain_flag":   retrain_check["retrain"],
            "reason":         retrain_check["reason"],
        }


class ReportingAgent:
    """
    Generates natural language summaries of batch results.
    Surfaces anomalies and performance alerts.
    """

    SYSTEM_PROMPT = """You are a Reporting Agent for an aerial imagery AI system.
Your job is to:
1. Generate clear, concise natural language summaries of batch processing results
2. Highlight anomalies and performance concerns
3. Provide actionable recommendations for operators

Write in a professional technical style. Be specific with numbers.
Structure your report with: Summary, Key Findings, Anomalies, Recommendations."""

    def __init__(self):
        self.llm = get_llm(temperature=0.3)

    def generate_batch_summary(
        self,
        batch_summary: Dict,
        qc_results: List[Dict],
        drift_status: Dict,
    ) -> str:
        """Generate a natural language summary of batch results."""

        flagged    = [r for r in qc_results if r.get("decision") != "accept"]
        reinfered  = [r for r in qc_results if r.get("decision") == "reinfer"]
        escalated  = [r for r in qc_results if r.get("decision") == "escalate"]
        failure_modes = []
        for r in qc_results:
            failure_modes.extend(r.get("failure_modes", []))

        prompt = f"""Generate a batch processing report for this aerial imagery pipeline run:

BATCH STATISTICS:
- Total images processed: {batch_summary.get('total_processed', 0)}
- Crop images: {batch_summary.get('crop_images', 0)}
- Traffic images: {batch_summary.get('traffic_images', 0)}
- Mean confidence: {batch_summary.get('mean_confidence', 0):.4f}
- Flagged for QC: {len(flagged)}
- Re-inferred: {len(reinfered)}
- Escalated: {len(escalated)}

QC FINDINGS:
- Failure modes detected: {list(set(failure_modes)) or 'None'}
- Drift detected: {drift_status.get('drift_detected', False)}
- Drift score: {drift_status.get('drift_score', 'N/A')}
- Retrain flag: {drift_status.get('retrain_flag', False)}

Generate a professional batch report with Summary, Key Findings, Anomalies, and Recommendations."""

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"""BATCH REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}

Summary:
Processed {batch_summary.get('total_processed', 0)} images
({batch_summary.get('crop_images', 0)} crop, {batch_summary.get('traffic_images', 0)} traffic).
Mean confidence: {batch_summary.get('mean_confidence', 0):.4f}.
{len(flagged)} images flagged by QC ({len(reinfered)} re-inferred, {len(escalated)} escalated).

Anomalies:
Drift detected: {drift_status.get('drift_detected', False)}.
{f"Drift score: {drift_status.get('drift_score')}" if drift_status.get('drift_score') else ''}

[Note: LLM summary unavailable ({e}). Rule-based report generated.]"""

    def generate_anomaly_alert(self, anomalies: List[Dict]) -> str:
        """Generate a focused alert for critical anomalies."""
        if not anomalies:
            return "No anomalies detected in this batch."

        prompt = f"""Generate a concise anomaly alert for these detections:
{json.dumps(anomalies, indent=2)}

Keep it under 100 words. Lead with severity. List affected images."""

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception:
            return (f"⚠ ANOMALY ALERT: {len(anomalies)} images flagged. "
                    f"Images: {[a.get('image_path', '?') for a in anomalies[:3]]}...")
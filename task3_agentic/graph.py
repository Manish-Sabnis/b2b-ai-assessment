import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).parent))
from agents.agents import OrchestratorAgent, QualityControlAgent, ReportingAgent
from tools.cv_tools import run_crop_pipeline, run_traffic_pipeline, run_reinference
from memory.memory import ShortTermMemory, LongTermMemory



class PipelineState(TypedDict):
    # Input
    image_paths:      List[str]
    batch_id:         str

    # Routing
    routing_decisions: List[Dict]

    # Inference
    inference_results: List[Dict]

    # QC
    qc_results:        List[Dict]
    flagged_images:    List[str]
    reprocessed:       List[Dict]

    # Drift
    drift_status:      Dict

    # Output
    batch_summary:     Dict
    report:            str
    anomaly_alert:     str
    logs:              List[str]


def ingest_batch(state: PipelineState) -> PipelineState:
    """Node 1: Validate and prepare image batch."""
    logs = state.get("logs", [])
    paths = state["image_paths"]
    batch_id = state.get("batch_id", f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    logs.append(f"[{datetime.now().isoformat()}] Ingesting batch {batch_id}: {len(paths)} images")
    print(f"\n{'='*60}")
    print(f"[Ingest] Batch {batch_id} — {len(paths)} images")
    print(f"{'='*60}")

    return {**state, "batch_id": batch_id, "logs": logs,
            "routing_decisions": [], "inference_results": [],
            "qc_results": [], "flagged_images": [], "reprocessed": [],
            "drift_status": {}, "batch_summary": {}, "report": "",
            "anomaly_alert": ""}


def route_images(state: PipelineState) -> PipelineState:
    """Node 2: Orchestrator Agent routes each image to correct pipeline."""
    logs = state.get("logs", [])
    orchestrator = OrchestratorAgent()
    decisions = orchestrator.schedule_batch(state["image_paths"])
    logs.append(f"[{datetime.now().isoformat()}] Routing complete: "
                f"{sum(1 for d in decisions if d['domain']=='crop')} crop, "
                f"{sum(1 for d in decisions if d['domain']=='traffic')} traffic")
    return {**state, "routing_decisions": decisions, "logs": logs}


def run_inference(state: PipelineState) -> PipelineState:
    """Node 3: Run CV pipelines based on routing decisions."""
    logs = state.get("logs", [])
    results = []
    print(f"\n[Inference] Running CV pipelines...")

    for decision in state["routing_decisions"]:
        path   = decision["image_path"]
        domain = decision["domain"]

        if domain == "crop":
            result = run_crop_pipeline(path, use_mock=True)
        else:
            result = run_traffic_pipeline(path, use_mock=True)

        results.append(result)
        conf = result.get("mean_confidence", 0)
        n    = result.get("n_polygons") or result.get("n_vehicles", 0)
        print(f"  ✓ {Path(path).name:<30} [{domain}] conf={conf:.3f} n={n}")

    logs.append(f"[{datetime.now().isoformat()}] Inference complete: {len(results)} results")
    return {**state, "inference_results": results, "logs": logs}


def quality_control(state: PipelineState) -> PipelineState:
    """Node 4: QC Agent validates results, triggers re-inference if needed."""
    logs = state.get("logs", [])
    qc_agent  = QualityControlAgent()
    qc_results  = []
    flagged     = []
    reprocessed = []

    print(f"\n[QC Agent] Validating {len(state['inference_results'])} results...")

    for result in state["inference_results"]:
        qc = qc_agent.validate(result)
        qc_results.append(qc)

        if qc["decision"] == "accept":
            print(f"  ✓ {Path(result['image_path']).name:<30} ACCEPTED  (score={qc['quality_score']:.3f})")

        elif qc["decision"] == "reinfer":
            path   = result["image_path"]
            domain = result["task"].replace("_detection", "")
            flagged.append(path)
            print(f"  ↻ {Path(path).name:<30} RE-INFERRING (conf={result['mean_confidence']:.3f})")
            logs.append(f"[QC] Re-inference triggered: {path} — {qc['reasoning']}")
            new_result = run_reinference(path, domain)
            new_qc     = qc_agent.validate(new_result)
            reprocessed.append(new_result)
            print(f"    → After re-inference: conf={new_result['mean_confidence']:.3f} "
                  f"decision={new_qc['decision']}")

        elif qc["decision"] == "escalate":
            path = result["image_path"]
            flagged.append(path)
            print(f"  ✗ {Path(path).name:<30} ESCALATED  (conf={result['mean_confidence']:.3f})")
            logs.append(f"[QC] Escalation: {path} — {qc['reasoning']}")

    logs.append(f"[{datetime.now().isoformat()}] QC complete: "
                f"{len(qc_results) - len(flagged)} accepted, "
                f"{len(reprocessed)} re-inferred, "
                f"{len(flagged) - len(reprocessed)} escalated")

    return {**state, "qc_results": qc_results, "flagged_images": flagged,
            "reprocessed": reprocessed, "logs": logs}


def check_drift(state: PipelineState) -> PipelineState:
    """Node 5: QC Agent checks cross-batch drift and updates long-term memory."""
    logs     = state.get("logs", [])
    lt_mem   = LongTermMemory()
    qc_agent = QualityControlAgent()
    results   = state["inference_results"]
    total     = len(results)
    mean_conf = sum(r.get("mean_confidence", 0) for r in results) / max(total, 1)

    batch_summary = {
        "batch_id":        state["batch_id"],
        "total_processed": total,
        "crop_images":     sum(1 for r in results if r.get("task") == "crop_detection"),
        "traffic_images":  sum(1 for r in results if r.get("task") == "traffic_detection"),
        "mean_confidence": round(mean_conf, 4),
        "flagged":         len(state["flagged_images"]),
        "reprocessed":     len(state["reprocessed"]),
    }
    lt_mem.log_batch(batch_summary)


    drift_status = qc_agent.check_drift(lt_mem)
    print(f"\n[Drift Check] score={drift_status.get('drift_score', 'N/A')} | "
          f"retrain={drift_status.get('retrain_flag', False)}")

    logs.append(f"[{datetime.now().isoformat()}] Drift check: {drift_status['reason']}")
    return {**state, "batch_summary": batch_summary,
            "drift_status": drift_status, "logs": logs}


def generate_report(state: PipelineState) -> PipelineState:
    """Node 6: Reporting Agent generates NL summary and anomaly alerts."""
    logs         = state.get("logs", [])
    reporter     = ReportingAgent()
    qc_results   = state["qc_results"]
    drift_status = state["drift_status"]

    print(f"\n[Reporting Agent] Generating batch report...")


    report = reporter.generate_batch_summary(
        state["batch_summary"], qc_results, drift_status
    )


    anomalies = [r for r in qc_results if r.get("decision") == "escalate"]
    alert     = reporter.generate_anomaly_alert(anomalies)


    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    batch_id = state["batch_id"]

    report_path = out_dir / f"{batch_id}_report.txt"
    log_path    = out_dir / f"{batch_id}_log.json"

    with open(report_path, "w") as f:
        f.write(f"BATCH REPORT — {batch_id}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        if anomalies:
            f.write("\n\n" + "=" * 60 + "\nANOMALY ALERT\n" + "=" * 60 + "\n")
            f.write(alert)

    with open(log_path, "w") as f:
        json.dump({
            "batch_id":     batch_id,
            "logs":         logs,
            "batch_summary": state["batch_summary"],
            "drift_status": drift_status,
            "qc_results":   qc_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(report[:600] + "..." if len(report) > 600 else report)
    print(f"\nReport saved → {report_path}")
    print(f"Log saved    → {log_path}")

    logs.append(f"[{datetime.now().isoformat()}] Report generated → {report_path}")
    return {**state, "report": report, "anomaly_alert": alert, "logs": logs}



def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("ingest_batch",    ingest_batch)
    graph.add_node("route_images",    route_images)
    graph.add_node("run_inference",   run_inference)
    graph.add_node("quality_control", quality_control)
    graph.add_node("check_drift",     check_drift)
    graph.add_node("generate_report", generate_report)
    graph.set_entry_point("ingest_batch")
    graph.add_edge("ingest_batch",    "route_images")
    graph.add_edge("route_images",    "run_inference")
    graph.add_edge("run_inference",   "quality_control")
    graph.add_edge("quality_control", "check_drift")
    graph.add_edge("check_drift",     "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


def run_pipeline(image_paths: List[str], batch_id: str = None) -> Dict:
    """Run the full agentic pipeline on a batch of images."""
    app = build_graph()

    initial_state = PipelineState(
        image_paths       = image_paths,
        batch_id          = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        routing_decisions = [],
        inference_results = [],
        qc_results        = [],
        flagged_images    = [],
        reprocessed       = [],
        drift_status      = {},
        batch_summary     = {},
        report            = "",
        anomaly_alert     = "",
        logs              = [],
    )

    final_state = app.invoke(initial_state)
    return final_state
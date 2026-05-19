import json, logging, traceback
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from tools.hitl_tool import HumanInTheLoopTool

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_DIR

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "agent_actions.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def log_action(agent: str, action: str, payload: dict):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "agent":     agent,
        "action":    action,
        "payload":   payload,
    }
    logging.info(json.dumps(entry))
    print(f"[LOG] {entry['timestamp']} | {agent} | {action}")

class DermaGuardOrchestrator:
    def __init__(self, classifier_agent, report_agent):
        self.classifier_agent = classifier_agent
        self.report_agent     = report_agent
        self.hitl_tool        = HumanInTheLoopTool()

    def run(self, image_path: str):
        log_action("Orchestrator", "pipeline_start", {"image_path": image_path})

        try:
            # --- Task 1: Classify ---
            classify_task = Task(
                description=f"Classify the skin lesion image at: {image_path}",
                expected_output=(
                    "A dict with keys: predicted_class, confidence, "
                    "probabilities, flag_for_review, image_path."
                ),
                agent=self.classifier_agent,
            )

            # --- Task 2: Generate report ---
            report_task = Task(
                description=(
                    "Using the classification results from the previous task, "
                    "generate a structured clinical triage report."
                ),
                expected_output="A complete clinical triage report as a formatted string.",
                agent=self.report_agent,
                context=[classify_task],
            )

            crew = Crew(
                agents=[self.classifier_agent, self.report_agent],
                tasks=[classify_task, report_task],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()
            log_action("Orchestrator", "crew_completed", {"result_preview": str(result)[:200]})

            # --- HITL checkpoint ---
            # Extract classification data from crew context
            classify_output = classify_task.output.raw if hasattr(classify_task, 'output') else {}
            if isinstance(classify_output, str):
                import ast
                try:
                    classify_output = ast.literal_eval(classify_output)
                except Exception:
                    classify_output = {}

            hitl_result = self.hitl_tool._run(
                report          = str(result),
                predicted_class = classify_output.get("predicted_class", "Unknown"),
                confidence      = classify_output.get("confidence", 0.0),
                image_path      = image_path,
            )
            log_action("Orchestrator", "hitl_decision", hitl_result)

            if hitl_result["status"] == "revise":
                print("\n→ Revision requested. Re-running report generation...\n")
                return self.run(image_path)

            return hitl_result

        except FileNotFoundError as e:
            log_action("Orchestrator", "error_file_not_found", {"error": str(e)})
            print(f"[ERROR] Image not found: {e}")
        except Exception as e:
            log_action("Orchestrator", "error_unexpected", {"error": traceback.format_exc()})
            print(f"[ERROR] Unexpected error: {e}")

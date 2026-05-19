import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.classifier_agent import build_classifier_agent
from agents.report_agent import build_report_agent
from agents.orchestrator import DermaGuardOrchestrator

def main():
    parser = argparse.ArgumentParser(description="DermaGuard — Skin Lesion Triage System")
    parser.add_argument("image_path", help="Path to the skin lesion image (JPEG or PNG)")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image_path)
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    print("\n" + "="*60)
    print("  DermaGuard — Multi-Agent Skin Lesion Triage")
    print("="*60 + "\n")

    classifier = build_classifier_agent()
    reporter   = build_report_agent()
    orchestrator = DermaGuardOrchestrator(classifier, reporter)

    result = orchestrator.run(image_path)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Status : {result.get('status', 'unknown')}")
    if result.get("output_path"):
        print(f"  Report : {result['output_path']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

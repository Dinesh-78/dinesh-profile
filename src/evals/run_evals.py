"""CLI Runner for AI Evaluations suite."""

import sys
import json
import logging
from pathlib import Path

from src.config import settings
from src.guardrails.pipeline import get_pipeline
from src.evals.evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_evaluation_suite():
    """Runs the full evaluation benchmark and exports results."""
    eval_dataset_path = Path(__file__).parent / "eval_dataset.json"
    if not eval_dataset_path.exists():
        logger.error(f"Evaluation dataset not found at {eval_dataset_path}")
        return None

    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n{'='*70}")
    print(f"🚀 STARTING DINESH PROFILE RAG & GUARDRAILS EVALUATION SUITE")
    print(f"Total Test Cases: {len(dataset)}")
    print(f"{'='*70}\n")

    pipeline = get_pipeline()
    evaluator = RAGEvaluator()

    report = evaluator.evaluate_all(
        test_dataset=dataset,
        pipeline_func=lambda q: pipeline.process(q).model_dump()
    )

    # Print summary table
    print(f"\n{'ID':<8} | {'CATEGORY':<25} | {'GUARDRAIL':<10} | {'FAITHFUL':<8} | {'RELEV':<8} | {'SCORE':<8}")
    print("-" * 75)
    for res in report.test_results:
        print(f"{res.test_id:<8} | {res.category:<25} | {res.guardrail_status:<10} | {res.faithfulness_score:<8.2f} | {res.answer_relevance_score:<8.2f} | {res.overall_score:<8.2f}")

    print("\n" + "="*70)
    print("📊 AGGREGATE BENCHMARK RESULTS")
    print("="*70)
    print(f"Total Tests Evaluated       : {report.total_tests}")
    print(f"Passed Tests                : {report.passed_tests}/{report.total_tests} ({round(report.passed_tests/report.total_tests*100, 1)}%)")
    print(f"Guardrail Accuracy / Defense: {round(report.guardrail_accuracy * 100, 1)}%")
    print(f"Average Faithfulness        : {round(report.avg_faithfulness * 100, 1)}%")
    print(f"Average Answer Relevance    : {round(report.avg_answer_relevance * 100, 1)}%")
    print(f"Average Context Recall      : {round(report.avg_context_recall * 100, 1)}%")
    print(f"Overall Composite Score     : {round(report.overall_benchmark_score * 100, 1)} / 100")
    print("="*70)

    # Save outputs to reports folder
    report_dir = Path(settings.CHROMA_PERSIST_DIR).parent / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    md_path = report_dir / "eval_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"""# RAG & Guardrails Evaluation Benchmark Report

## Overall Scores
- **Overall Composite Benchmark**: **{round(report.overall_benchmark_score * 100, 1)}%**
- **Guardrail Interception Accuracy**: **{round(report.guardrail_accuracy * 100, 1)}%**
- **Average Faithfulness / Groundedness**: **{round(report.avg_faithfulness * 100, 1)}%**
- **Average Answer Relevance**: **{round(report.avg_answer_relevance * 100, 1)}%**
- **Average Context Recall**: **{round(report.avg_context_recall * 100, 1)}%**
- **Total Tests Passed**: {report.passed_tests} / {report.total_tests}

## Category Breakdown
""")
        for cat, data in report.category_breakdown.items():
            f.write(f"- **{cat}**: Average Score = {round(data['avg_score'] * 100, 1)}% (Tests: {data['count']})\n")

        f.write("\n## Detailed Test Results\n\n")
        f.write("| Test ID | Category | Guardrail Action | Faithfulness | Relevancy | Overall Score | Notes |\n")
        f.write("|---------|----------|------------------|--------------|-----------|---------------|-------|\n")
        for r in report.test_results:
            f.write(f"| {r.test_id} | {r.category} | {r.guardrail_status} | {r.faithfulness_score:.2f} | {r.answer_relevance_score:.2f} | {r.overall_score:.2f} | {r.notes} |\n")

    logger.info(f"Evaluation report written to {json_path} and {md_path}")
    return report

if __name__ == "__main__":
    run_evaluation_suite()


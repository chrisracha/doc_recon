#!/usr/bin/env python
"""
Evaluation script for the Document Reconstruction Pipeline.

Computes metrics on processed documents and compares against expected outputs.

Usage:
    python eval.py --input <json_output> [--expected <expected_json>]
    python eval.py --results-dir <dir> --report <report.json>
"""

import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a processed document."""
    # Overall metrics
    global_confidence: float = 0.0
    coverage_pct: float = 0.0
    
    # Text metrics
    text_blocks_total: int = 0
    text_blocks_high_confidence: int = 0
    text_blocks_low_confidence: int = 0
    text_confidence_avg: float = 0.0
    
    # Equation metrics
    equations_total: int = 0
    equations_valid: int = 0
    equations_invalid: int = 0
    equation_success_rate: float = 0.0
    
    # Table metrics
    tables_total: int = 0
    tables_reconstructed: int = 0
    tables_partial: int = 0
    table_success_rate: float = 0.0
    
    # Accuracy (if expected output provided)
    text_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_document(json_path: Path) -> Dict[str, Any]:
    """Load a document JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_document(doc: Dict[str, Any]) -> EvaluationMetrics:
    """Evaluate a single processed document."""
    metrics = EvaluationMetrics()
    
    # Get pre-computed metrics if available
    doc_metrics = doc.get("metrics", {})
    metrics.global_confidence = doc_metrics.get("global_confidence", 0.0)
    metrics.coverage_pct = doc_metrics.get("coverage_pct", 0.0)
    
    # Analyze blocks
    text_confidences = []
    
    for page in doc.get("pages", []):
        for block in page.get("blocks", []):
            block_type = block.get("type", "")
            confidence = block.get("confidence", 0.0)
            
            # Text blocks
            if block.get("text") is not None:
                metrics.text_blocks_total += 1
                text_confidences.append(confidence)
                
                if confidence >= 0.80:
                    metrics.text_blocks_high_confidence += 1
                elif confidence < 0.65:
                    metrics.text_blocks_low_confidence += 1
            
            # Equation blocks
            if block.get("latex") is not None:
                metrics.equations_total += 1
                latex = block.get("latex", "")
                
                # Simple validity check: length > 10 and non-empty
                if len(latex) > 10 and latex.strip():
                    metrics.equations_valid += 1
                else:
                    metrics.equations_invalid += 1
            
            # Table blocks
            if block.get("table") is not None:
                metrics.tables_total += 1
                table = block.get("table", {})
                status = table.get("status", "unknown")
                
                if status == "success":
                    metrics.tables_reconstructed += 1
                elif status == "partial":
                    metrics.tables_partial += 1
    
    # Calculate averages and rates
    if text_confidences:
        metrics.text_confidence_avg = sum(text_confidences) / len(text_confidences)
    
    if metrics.equations_total > 0:
        metrics.equation_success_rate = metrics.equations_valid / metrics.equations_total
    
    if metrics.tables_total > 0:
        metrics.table_success_rate = (
            (metrics.tables_reconstructed + 0.5 * metrics.tables_partial) /
            metrics.tables_total
        )
    
    return metrics


def compare_text(expected: str, actual: str) -> float:
    """
    Compare expected and actual text, return similarity score.
    Uses simple word-level Jaccard similarity.
    """
    if not expected or not actual:
        return 0.0
    
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    
    if not expected_words:
        return 0.0
    
    intersection = expected_words & actual_words
    union = expected_words | actual_words
    
    return len(intersection) / len(union) if union else 0.0


def evaluate_against_expected(
    doc: Dict[str, Any],
    expected: Dict[str, Any]
) -> EvaluationMetrics:
    """Evaluate document against expected output."""
    metrics = evaluate_document(doc)
    
    # Compare text content
    actual_text = doc.get("markdown", "")
    expected_text = expected.get("markdown", "")
    
    if actual_text and expected_text:
        metrics.text_accuracy = compare_text(expected_text, actual_text)
    
    return metrics


def print_metrics(metrics: EvaluationMetrics, name: str = "Document"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {name}")
    print('='*60)
    
    print("\n📊 Overall Metrics:")
    print(f"  Global Confidence: {metrics.global_confidence:.1%}")
    print(f"  Page Coverage: {metrics.coverage_pct:.1f}%")
    
    print("\n📝 Text Blocks:")
    print(f"  Total: {metrics.text_blocks_total}")
    print(f"  High Confidence (≥80%): {metrics.text_blocks_high_confidence}")
    print(f"  Low Confidence (<65%): {metrics.text_blocks_low_confidence}")
    print(f"  Average Confidence: {metrics.text_confidence_avg:.1%}")
    
    print("\n🔢 Equations:")
    print(f"  Total: {metrics.equations_total}")
    print(f"  Valid (length > 10): {metrics.equations_valid}")
    print(f"  Invalid: {metrics.equations_invalid}")
    print(f"  Success Rate: {metrics.equation_success_rate:.1%}")
    
    print("\n📋 Tables:")
    print(f"  Total: {metrics.tables_total}")
    print(f"  Fully Reconstructed: {metrics.tables_reconstructed}")
    print(f"  Partial: {metrics.tables_partial}")
    print(f"  Success Rate: {metrics.table_success_rate:.1%}")
    
    if metrics.text_accuracy is not None:
        print("\n🎯 Accuracy (vs expected):")
        print(f"  Text Similarity: {metrics.text_accuracy:.1%}")
    
    print('='*60)


def evaluate_directory(
    results_dir: Path,
    expected_dir: Optional[Path] = None
) -> Dict[str, EvaluationMetrics]:
    """Evaluate all documents in a directory."""
    results = {}
    
    for json_file in results_dir.glob("*.json"):
        if json_file.name == "report.json":
            continue
        
        doc = load_document(json_file)
        
        # Check for expected output
        expected = None
        if expected_dir:
            expected_file = expected_dir / json_file.name
            if expected_file.exists():
                expected = load_document(expected_file)
        
        if expected:
            metrics = evaluate_against_expected(doc, expected)
        else:
            metrics = evaluate_document(doc)
        
        results[json_file.stem] = metrics
    
    return results


def generate_report(
    results: Dict[str, EvaluationMetrics]
) -> Dict[str, Any]:
    """Generate a summary report from multiple evaluations."""
    if not results:
        return {"error": "No results to report"}
    
    # Aggregate metrics
    total_docs = len(results)
    
    avg_confidence = sum(m.global_confidence for m in results.values()) / total_docs
    avg_coverage = sum(m.coverage_pct for m in results.values()) / total_docs
    
    total_text = sum(m.text_blocks_total for m in results.values())
    total_equations = sum(m.equations_total for m in results.values())
    total_tables = sum(m.tables_total for m in results.values())
    
    total_eq_valid = sum(m.equations_valid for m in results.values())
    total_tables_ok = sum(m.tables_reconstructed for m in results.values())
    
    return {
        "summary": {
            "documents_evaluated": total_docs,
            "average_confidence": round(avg_confidence, 3),
            "average_coverage_pct": round(avg_coverage, 1),
            "total_text_blocks": total_text,
            "total_equations": total_equations,
            "total_tables": total_tables,
            "equation_success_rate": round(total_eq_valid / total_equations, 3) if total_equations > 0 else 0,
            "table_success_rate": round(total_tables_ok / total_tables, 3) if total_tables > 0 else 0
        },
        "individual_results": {
            name: metrics.to_dict()
            for name, metrics in results.items()
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Document Reconstruction outputs"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Path to document JSON output file"
    )
    
    parser.add_argument(
        "--expected", "-e",
        type=Path,
        help="Path to expected output JSON for comparison"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing multiple result JSON files"
    )
    
    parser.add_argument(
        "--expected-dir",
        type=Path,
        help="Directory containing expected outputs"
    )
    
    parser.add_argument(
        "--report", "-r",
        type=Path,
        help="Output path for evaluation report JSON"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress printed output"
    )
    
    args = parser.parse_args()
    
    results = {}
    
    # Evaluate single file
    if args.input:
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        doc = load_document(args.input)
        
        if args.expected and args.expected.exists():
            expected = load_document(args.expected)
            metrics = evaluate_against_expected(doc, expected)
        else:
            metrics = evaluate_document(doc)
        
        results[args.input.stem] = metrics
        
        if not args.quiet:
            print_metrics(metrics, args.input.name)
    
    # Evaluate directory
    elif args.results_dir:
        if not args.results_dir.is_dir():
            logger.error(f"Results directory not found: {args.results_dir}")
            sys.exit(1)
        
        results = evaluate_directory(args.results_dir, args.expected_dir)
        
        if not args.quiet:
            for name, metrics in results.items():
                print_metrics(metrics, name)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    # Generate and save report
    if args.report and results:
        report = generate_report(results)
        
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {args.report}")
        
        if not args.quiet:
            print(f"\n📝 Report saved to: {args.report}")
            print("\nSummary:")
            for key, value in report["summary"].items():
                print(f"  {key}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arwen_etl.ocr import ocr_processor
from arwen_etl.whisper_asr import create_asr_processor
from arwen_etl.diarization import SpeakerDiarizer
from arwen_etl.multilingual import language_detector, translator
from arwen_etl.video_processor import FFmpegVideoProcessor
from arwen_etl.policy_inference import PolicyExtractor
from arwen_etl.org_resolution import org_resolver
from arwen_benchmark.benchmark_cases import (
    OCRBenchmarkCase,
    ASRBenchmarkCase,
    DiarizationBenchmarkCase,
    MultilingualBenchmarkCase,
    StakeholderBenchmarkCase,
    PositionBenchmarkCase,
    ArgumentBenchmarkCase,
    EvidenceLinkingBenchmarkCase,
    TemporalTrackingBenchmarkCase,
)
from arwen_benchmark.metrics import reciprocal_rank, evidence_recall, coverage_score


class BenchmarkRunner:
    """Run benchmarks against implemented components."""

    def __init__(self, data_dir: str = "tests/benchmark_data"):
        self.data_dir = Path(data_dir)
        self.asr_processor = create_asr_processor()
        self.diarizer = SpeakerDiarizer()
        self.video_processor = FFmpegVideoProcessor()
        self.policy_extractor = PolicyExtractor()
        self.ocr_processor = ocr_processor
        self.language_detector = language_detector
        self.translator = translator
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def run_ocr_benchmarks(self) -> List[Dict[str, Any]]:
        """Run OCR benchmark cases."""
        cases_file = self.data_dir / "ocr_cases.json"
        if not cases_file.exists():
            return []

        with open(cases_file) as f:
            cases_data = json.load(f)

        results = []
        for case_dict in cases_data:
            case = OCRBenchmarkCase(**case_dict)
            try:
                result = self.ocr_processor.process_image(case.image_path)
                # Simple exact match (could use fuzzy matching)
                exact_match = result["text"].strip() == case.expected_text.strip()
                confidence_ok = result["confidence"] >= case.expected_confidence_min
                results.append({
                    "case_id": case.case_id,
                    "expected": case.expected_text,
                    "got": result["text"],
                    "exact_match": exact_match,
                    "confidence": result["confidence"],
                    "confidence_ok": confidence_ok,
                    "passed": exact_match and confidence_ok,
                })
            except Exception as e:
                results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "passed": False,
                })
        self.results["ocr"] = results
        return results

    def run_asr_benchmarks(self) -> List[Dict[str, Any]]:
        """Run ASR benchmark cases."""
        cases_file = self.data_dir / "asr_cases.json"
        if not cases_file.exists():
            return []

        with open(cases_file) as f:
            cases_data = json.load(f)

        results = []
        for case_dict in cases_data:
            case = ASRBenchmarkCase(**case_dict)
            try:
                result = self.asr_processor.transcribe(case.audio_path, language=case.expected_language[:2] if case.expected_language else None)
                # Simple WER (word error rate) approximation
                expected_words = set(case.expected_transcript.lower().split())
                got_words = set(result["text"].lower().split())
                if not expected_words:
                    wer = 1.0 if got_words else 0.0
                else:
                    wer = 1.0 - len(expected_words & got_words) / len(expected_words)
                results.append({
                    "case_id": case.case_id,
                    "expected": case.expected_transcript,
                    "got": result["text"],
                    "wer": wer,
                    "passed": wer <= 0.3,  # 30% WER threshold
                })
            except Exception as e:
                results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "passed": False,
                })
        self.results["asr"] = results
        return results

    def run_diarization_benchmarks(self) -> List[Dict[str, Any]]:
        """Run speaker diarization benchmark cases."""
        cases_file = self.data_dir / "diarization_cases.json"
        if not cases_file.exists():
            return []

        with open(cases_file) as f:
            cases_data = json.load(f)

        results = []
        for case_dict in cases_data:
            case = DiarizationBenchmarkCase(**case_dict)
            try:
                segments = self.diarizer.diarize(case.audio_path)
                # Simple overlap metric
                expected = [(s["start"], s["end"]) for s in case.expected_segments]
                got = [(s["start"], s["end"]) for s in segments]
                # Compute IoU-like score
                total_overlap = 0.0
                total_expected = sum(e[1] - e[0] for e in expected)
                if total_expected > 0:
                    for exp_start, exp_end in expected:
                        for got_start, got_end in got:
                            overlap = max(0, min(exp_end, got_end) - max(exp_start, got_start))
                            total_overlap += overlap
                    score = total_overlap / total_expected if total_expected > 0 else 0.0
                else:
                    score = 0.0
                results.append({
                    "case_id": case.case_id,
                    "expected_segments": case.expected_segments,
                    "got_segments": segments,
                    "overlap_score": score,
                    "passed": score >= 0.5,
                })
            except Exception as e:
                results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "passed": False,
                })
        self.results["diarization"] = results
        return results

    def run_multilingual_benchmarks(self) -> List[Dict[str, Any]]:
        """Run multilingual benchmark cases."""
        cases_file = self.data_dir / "multilingual_cases.json"
        if not cases_file.exists():
            return []

        with open(cases_file) as f:
            cases_data = json.load(f)

        results = []
        for case_dict in cases_data:
            case = MultilingualBenchmarkCase(**case_dict)
            try:
                detected = self.language_detector.detect(case.text)
                supported = self.language_detector.is_supported(detected)
                results.append({
                    "case_id": case.case_id,
                    "text": case.text[:50],
                    "expected_language": case.expected_language,
                    "detected_language": detected,
                    "expected_supported": case.expected_supported,
                    "detected_supported": supported,
                    "passed": detected == case.expected_language and supported == case.expected_supported,
                })
            except Exception as e:
                results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "passed": False,
                })
        self.results["multilingual"] = results
        return results

    def run_stakeholder_benchmarks(self) -> List[Dict[str, Any]]:
        """Run stakeholder extraction benchmark cases."""
        cases_file = self.data_dir / "stakeholder_cases.json"
        if not cases_file.exists():
            return []

        with open(cases_file) as f:
            cases_data = json.load(f)

        results = []
        for case_dict in cases_data:
            case = StakeholderBenchmarkCase(**case_dict)
            try:
                # Use policy_extractor for stakeholder extraction (simplified)
                stakeholders, _ = self.policy_extractor.stakeholder_extraction(case.text)
                # Convert to comparable format
                got_names = set(stakeholders.keys())
                expected_names = set(s["name"] for s in case.expected_stakeholders)
                # Simple coverage
                if expected_names:
                    coverage = len(got_names & expected_names) / len(expected_names)
                else:
                    coverage = 1.0 if not got_names else 0.0
                results.append({
                    "case_id": case.case_id,
                    "expected": [s["name"] for s in case.expected_stakeholders],
                    "got": list(got_names),
                    "coverage": coverage,
                    "passed": coverage >= 0.5,
                })
            except Exception as e:
                results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "passed": False,
                })
        self.results["stakeholder"] = results
        return results

    def run_position_benchmarks(self) -> List[Dict[str, Any]]:
        """Run position extraction benchmark cases."""
        cases_file = self.data_dir / "position_cases.json"
        if not cases_file.exists():
            return []

        with open(cases_file) as f:
            cases_data = json.load(f)

        results = []
        for case_dict in cases_data:
            case = PositionBenchmarkCase(**case_dict)
            try:
                positions = self.policy_extractor.extract_positions(case.text)
                # Simple check if any position extracted
                got_any = len(positions) > 0
                expected_any = len(case.expected_positions) > 0
                results.append({
                    "case_id": case.case_id,
                    "expected_positions": case.expected_positions,
                    "got_positions": positions,
                    "got_any": got_any,
                    "expected_any": expected_any,
                    "passed": got_any == expected_any,  # At least detect if expected
                })
            except Exception as e:
                results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "passed": False,
                })
        self.results["position"] = results
        return results

    def run_argument_benchmarks(self) -> List[Dict[str, Any]]:
        """Run argument extraction benchmark cases."""
        cases_file = self.data_dir / "argument_cases.json"
        if not cases_file.exists():
            return []

        with open(cases_file) as f:
            cases_data = json.load(f)

        results = []
        for case_dict in cases_data:
            case = ArgumentBenchmarkCase(**case_dict)
            try:
                arguments = self.policy_extractor.argument_extraction(case.text)
                # Simple count comparison
                got_count = len(arguments)
                expected_count = len(case.expected_arguments)
                results.append({
                    "case_id": case.case_id,
                    "expected_count": expected_count,
                    "got_count": got_count,
                    "passed": got_count > 0 if expected_count > 0 else got_count == 0,
                })
            except Exception as e:
                results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "passed": False,
                })
        self.results["argument"] = results
        return results

    def run_evidence_linking_benchmarks(self) -> List[Dict[str, Any]]:
        """Run evidence linking benchmark cases."""
        cases_file = self.data_dir / "evidence_cases.json"
        if not cases_file.exists():
            return []

        with open(cases_file) as f:
            cases_data = json.load(f)

        results = []
        for case_dict in cases_data:
            case = EvidenceLinkingBenchmarkCase(**case_dict)
            try:
                # Compute similarity for each evidence candidate
                scores = []
                for evidence in case.evidence_candidates:
                    score = self.policy_extractor.evidence_linking(case.claim, evidence)
                    scores.append(score)
                # Get top matches
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:len(case.expected_relevant)]
                # Check if expected relevant are in top
                expected_set = set(case.expected_relevant)
                got_set = set(top_indices)
                if expected_set:
                    recall = len(expected_set & got_set) / len(expected_set)
                else:
                    recall = 1.0
                results.append({
                    "case_id": case.case_id,
                    "claim": case.claim,
                    "evidence_candidates": case.evidence_candidates,
                    "scores": scores,
                    "expected_relevant": case.expected_relevant,
                    "got_top_indices": top_indices,
                    "recall": recall,
                    "passed": recall >= 0.5,
                })
            except Exception as e:
                results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "passed": False,
                })
        self.results["evidence_linking"] = results
        return results

    def run_temporal_tracking_benchmarks(self) -> List[Dict[str, Any]]:
        """Run temporal tracking benchmark cases."""
        cases_file = self.data_dir / "temporal_cases.json"
        if not cases_file.exists():
            return []

        with open(cases_file) as f:
            cases_data = json.load(f)

        results = []
        for case_dict in cases_data:
            case = TemporalTrackingBenchmarkCase(**case_dict)
            try:
                temporal_positions = self.policy_extractor.temporal_position_tracking(case.text)
                # Simple check if any temporal positions extracted
                got_any = len(temporal_positions) > 0
                expected_any = len(case.expected_events) > 0
                results.append({
                    "case_id": case.case_id,
                    "expected_events": case.expected_events,
                    "got_temporal": temporal_positions,
                    "got_any": got_any,
                    "expected_any": expected_any,
                    "passed": got_any == expected_any,
                })
            except Exception as e:
                results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "passed": False,
                })
        self.results["temporal_tracking"] = results
        return results

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites."""
        start_time = time.time()
        self.results = {}

        self.run_ocr_benchmarks()
        self.run_asr_benchmarks()
        self.run_diarization_benchmarks()
        self.run_multilingual_benchmarks()
        self.run_stakeholder_benchmarks()
        self.run_position_benchmarks()
        self.run_argument_benchmarks()
        self.run_evidence_linking_benchmarks()
        self.run_temporal_tracking_benchmarks()

        end_time = time.time()

        # Summary
        summary = {
            "total_time_seconds": end_time - start_time,
            "suites": {},
        }

        for suite_name, results in self.results.items():
            if results:
                passed = sum(1 for r in results if r.get("passed", False))
                total = len(results)
                summary["suites"][suite_name] = {
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "pass_rate": passed / total if total > 0 else 0.0,
                }

        return summary

    def save_results(self, output_path: str = "tests/benchmark_results.json") -> None:
        """Save benchmark results to JSON."""
        results = {
            "timestamp": time.time(),
            "results": self.results,
            "summary": self.run_all_benchmarks()["suites"] if self.results else {},
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)


def main():
    """Run benchmark suite."""
    runner = BenchmarkRunner()
    summary = runner.run_all_benchmarks()
    print(json.dumps(summary, indent=2))
    runner.save_results()


if __name__ == "__main__":
    main()
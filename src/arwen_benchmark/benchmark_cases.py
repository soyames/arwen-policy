from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime


@dataclass(frozen=True)
class OCRBenchmarkCase:
    case_id: str
    image_path: str
    expected_text: str
    expected_language: str = "eng"
    expected_confidence_min: float = 0.8


@dataclass(frozen=True)
class ASRBenchmarkCase:
    case_id: str
    audio_path: str
    expected_transcript: str
    expected_language: str = "eng"
    expected_speakers: int = 1


@dataclass(frozen=True)
class DiarizationBenchmarkCase:
    case_id: str
    audio_path: str
    expected_segments: List[Dict[str, Any]]  # [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}, ...]


@dataclass(frozen=True)
class MultilingualBenchmarkCase:
    case_id: str
    text: str
    expected_language: str
    expected_supported: bool


@dataclass(frozen=True)
class StakeholderBenchmarkCase:
    case_id: str
    text: str
    expected_stakeholders: List[Dict[str, Any]]  # [{"name": "ICANN", "type": "organization", "role": "regulator"}, ...]


@dataclass(frozen=True)
class PositionBenchmarkCase:
    case_id: str
    text: str
    expected_positions: List[Dict[str, Any]]  # [{"position": "supports", "subject": "ICANN", "confidence_min": 0.8}, ...]


@dataclass(frozen=True)
class ArgumentBenchmarkCase:
    case_id: str
    text: str
    expected_arguments: List[Dict[str, Any]]  # [{"claim": "ICANN should", "justification": "because"}, ...]


@dataclass(frozen=True)
class EvidenceLinkingBenchmarkCase:
    case_id: str
    claim: str
    evidence_candidates: List[str]
    expected_relevant: List[int]  # Indices of relevant evidence


@dataclass(frozen=True)
class TemporalTrackingBenchmarkCase:
    case_id: str
    text: str
    expected_events: List[Dict[str, Any]]  # [{"text": "position X", "temporal_marker": "before", "reference_to": "position Y"}, ...]


class BenchmarkDataGenerator:
    """Generate synthetic benchmark data for testing."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def generate_ocr_cases(self, num_cases: int = 10) -> List[OCRBenchmarkCase]:
        """Generate OCR benchmark cases with known text."""
        cases = []
        sample_texts = [
            "ICANN Policy Update: The Board adopted the new gTLD agreement on June 15, 2023.",
            "IGF 2023 Session Transcript: Multi-stakeholder approaches to Internet governance.",
            "IETF RFC 9000: QUIC - A UDP-Based Multiplexed and Secure Transport.",
            "ITU Recommendation X.1500: Cybersecurity information exchange.",
            "UN Resolution A/RES/76/183: The role of the UN in promoting digital cooperation.",
        ]

        for i in range(num_cases):
            text = random.choice(sample_texts)
            cases.append(OCRBenchmarkCase(
                case_id=f"ocr_case_{i:03d}",
                image_path=f"test_data/ocr/sample_{i:03d}.png",  # Will need actual test images
                expected_text=text,
                expected_language="eng",
                expected_confidence_min=0.75
            ))
        return cases

    def generate_asr_cases(self, num_cases: int = 10) -> List[ASRBenchmarkCase]:
        """Generate ASR benchmark cases with known transcripts."""
        cases = []
        sample_transcripts = [
            "The Internet Governance Forum brings together stakeholders from around the world.",
            "ICANN manages the domain name system and IP address allocation.",
            "The IETF develops Internet standards through a consensus process.",
            "UNESCO promotes education, science, and culture worldwide.",
        ]

        for i in range(num_cases):
            transcript = random.choice(sample_transcripts)
            cases.append(ASRBenchmarkCase(
                case_id=f"asr_case_{i:03d}",
                audio_path=f"test_data/asr/sample_{i:03d}.wav",
                expected_transcript=transcript,
                expected_language="eng",
                expected_speakers=random.randint(1, 3)
            ))
        return cases

    def generate_stakeholder_cases(self, num_cases: int = 10) -> List[StakeholderBenchmarkCase]:
        """Generate stakeholder extraction benchmark cases."""
        cases = []
        sample_texts = [
            "ICANN, the Internet Corporation for Assigned Names and Numbers, announced new policies.",
            "The Internet Governance Forum (IGF) held its annual meeting in Kyoto.",
            "IETF Working Group chairs discussed draft standards for HTTP/3.",
            "The United Nations Department of Economic and Social Affairs (UN DESA) published a report.",
        ]

        for i in range(num_cases):
            text = random.choice(sample_texts)
            cases.append(StakeholderBenchmarkCase(
                case_id=f"stakeholder_case_{i:03d}",
                text=text,
                expected_stakeholders=[
                    {"name": "ICANN", "type": "organization", "role": "regulator"},
                    {"name": "IGF", "type": "organization", "role": "forum"},
                    {"name": "IETF", "type": "organization", "role": "standards_body"},
                    {"name": "UN DESA", "type": "organization", "role": "intergovernmental"},
                ]
            ))
        return cases

    def generate_position_cases(self, num_cases: int = 10) -> List[PositionBenchmarkCase]:
        """Generate position extraction benchmark cases."""
        cases = []
        sample_texts = [
            "ICANN should support the new gTLD program because it promotes competition.",
            "The IGF must ensure multi-stakeholder participation in all sessions.",
            "IETF should adopt the new transport protocol as it improves performance.",
            "UNESCO may consider expanding its digital literacy programs worldwide.",
        ]

        for i in range(num_cases):
            text = random.choice(sample_texts)
            cases.append(PositionBenchmarkCase(
                case_id=f"position_case_{i:03d}",
                text=text,
                expected_positions=[
                    {"position": "support", "subject": "ICANN", "confidence_min": 0.8},
                    {"position": "require", "subject": "IGF", "confidence_min": 0.85},
                    {"position": "recommend", "subject": "IETF", "confidence_min": 0.75},
                    {"position": "consider", "subject": "UNESCO", "confidence_min": 0.7},
                ]
            ))
        return cases

    def generate_argument_cases(self, num_cases: int = 10) -> List[ArgumentBenchmarkCase]:
        """Generate argument extraction benchmark cases."""
        cases = []
        sample_texts = [
            "ICANN should support the new gTLD program because it promotes competition and innovation.",
            "Although some oppose the changes, the IGF must ensure participation because diverse voices improve outcomes.",
            "IETF should adopt QUIC since it provides better performance than TCP.",
        ]

        for i in range(num_cases):
            text = random.choice(sample_texts)
            cases.append(ArgumentBenchmarkCase(
                case_id=f"argument_case_{i:03d}",
                text=text,
                expected_arguments=[
                    {"claim": "ICANN should support gTLD", "justification": "promotes competition"},
                    {"claim": "IGF must ensure participation", "justification": "diverse voices improve outcomes"},
                    {"claim": "IETF should adopt QUIC", "justification": "better performance than TCP"},
                ]
            ))
        return cases

    def generate_evidence_cases(self, num_cases: int = 10) -> List[EvidenceLinkingBenchmarkCase]:
        """Generate evidence linking benchmark cases."""
        cases = []
        claims = [
            "ICANN supports multi-stakeholder governance",
            "IGF promotes inclusive participation",
            "IETF develops open standards",
        ]
        evidence_sets = [
            [
                "ICANN's bylaws require multi-stakeholder model",
                "IGF charter mandates inclusive participation",
                "IETF RFC process is open to all",
                "UN resolution on digital cooperation",
            ],
            [
                "ICANN Board minutes from 2023",
                "IGF annual report 2023",
                "IETF meeting proceedings",
                "OECD digital policy framework",
            ]
        ]

        for i in range(num_cases):
            claim = random.choice(claims)
            evidence = random.choice(evidence_sets)
            cases.append(EvidenceLinkingBenchmarkCase(
                case_id=f"evidence_case_{i:03d}",
                claim=claim,
                evidence_candidates=evidence,
                expected_relevant=[0, 1, 2]  # First three are typically relevant
            ))
        return cases

    def generate_temporal_cases(self, num_cases: int = 10) -> List[TemporalTrackingBenchmarkCase]:
        """Generate temporal tracking benchmark cases."""
        cases = []
        sample_texts = [
            "Before 2020, ICANN opposed the proposal. Subsequently, ICANN changed its position and now supports it.",
            "Initially, IGF focused only on access issues. Later, it expanded to include security and governance.",
            "First, IETF developed TCP. Then it developed QUIC as a modern replacement.",
        ]

        for i in range(num_cases):
            text = random.choice(sample_texts)
            cases.append(TemporalTrackingBenchmarkCase(
                case_id=f"temporal_case_{i:03d}",
                text=text,
                expected_events=[
                    {"text": "ICANN opposed", "temporal_marker": "before", "reference_to": "ICANN supports"},
                    {"text": "ICANN supports", "temporal_marker": "subsequently", "reference_to": "ICANN opposed"},
                ]
            ))
        return cases


def load_benchmark_cases(benchmark_dir: str = "tests/benchmark_data") -> Dict[str, List[Any]]:
    """Load benchmark cases from JSON files."""
    cases = {}
    benchmark_path = Path(benchmark_dir)

    for case_file in benchmark_path.glob("*.json"):
        with case_file.open() as f:
            data = json.load(f)
            cases[case_file.stem] = data

    return cases


def save_benchmark_cases(cases: Dict[str, List[Any]], benchmark_dir: str = "tests/benchmark_data") -> None:
    """Save benchmark cases to JSON files."""
    benchmark_path = Path(benchmark_dir)
    benchmark_path.mkdir(parents=True, exist_ok=True)

    for case_type, case_list in cases.items():
        # Convert dataclasses to dicts
        serializable = []
        for case in case_list:
            if hasattr(case, '__dataclass_fields__'):
                serializable.append({k: v for k, v in case.__dict__.items()})
            else:
                serializable.append(case)

        output_file = benchmark_path / f"{case_type}.json"
        with output_file.open('w') as f:
            json.dump(serializable, f, indent=2)


def generate_all_benchmarks(output_dir: str = "tests/benchmark_data") -> Dict[str, List[Any]]:
    """Generate all benchmark cases and save them."""
    generator = BenchmarkDataGenerator()

    cases = {
        "ocr_cases": generator.generate_ocr_cases(10),
        "asr_cases": generator.generate_asr_cases(10),
        "stakeholder_cases": generator.generate_stakeholder_cases(10),
        "position_cases": generator.generate_position_cases(10),
        "argument_cases": generator.generate_argument_cases(10),
        "evidence_cases": generator.generate_evidence_cases(10),
        "temporal_cases": generator.generate_temporal_cases(10),
    }

    save_benchmark_cases(cases, output_dir)
    return cases


# Public API
benchmark_generator = BenchmarkDataGenerator()
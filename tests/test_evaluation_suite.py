"""Structural evaluation suite for Arwen Policy.

Tests prompts, configuration, generation pipeline, and dataset quality.
These do NOT require a GPU or model inference.
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest

DATA_DIR = Path("datasets/sft_final")


class TestCanonicalPrompt:
    def test_prompt_importable(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT, get_system_prompt
        assert len(ARWEN_SYSTEM_PROMPT) > 5000
        assert "multistakeholder" in ARWEN_SYSTEM_PROMPT.lower()
        assert get_system_prompt() == ARWEN_SYSTEM_PROMPT

    def test_rejects_source_only(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        assert "only the supplied source evidence" not in ARWEN_SYSTEM_PROMPT

    def test_evidence_not_prerequisite(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        assert "not a prerequisite for policy reasoning" in ARWEN_SYSTEM_PROMPT.lower()

    def test_distinguishes_perspective_from_position(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        p = ARWEN_SYSTEM_PROMPT.lower()
        assert "general stakeholder perspective" in p
        assert "documented" in p

    def test_defines_question_types(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        p = ARWEN_SYSTEM_PROMPT.lower()
        assert "general policy question" in p
        assert "source-supported" in p
        assert "source-specific" in p

    def test_has_hriam(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        assert "HRIAM" in ARWEN_SYSTEM_PROMPT
        assert "Human Rights Impact Assessment" in ARWEN_SYSTEM_PROMPT

    def test_has_trigger_states(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        assert "HRIAM_NOT_MATERIAL" in ARWEN_SYSTEM_PROMPT
        assert "HRIAM_RELEVANT" in ARWEN_SYSTEM_PROMPT
        assert "HRIAM_CENTRAL" in ARWEN_SYSTEM_PROMPT

    def test_distinguishes_impact_from_violation(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        p = ARWEN_SYSTEM_PROMPT.lower()
        assert "potential impact" in p
        assert "violation" in p

    def test_has_duty_bearers(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        assert "duty-bearer" in ARWEN_SYSTEM_PROMPT.lower()

    def test_security_not_standalone_right(self):
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        p = ARWEN_SYSTEM_PROMPT.lower()
        assert "not itself a standalone human right" in p or "not a standalone human right" in p

    def test_scripts_no_old_prompt(self):
        for script in ["scripts/train_qlora.py", "scripts/eval_test_set.py",
                       "scripts/verify_adapter.py"]:
            if not Path(script).exists():
                continue
            content = Path(script).read_text(encoding="utf-8")
            assert "only the supplied source evidence" not in content

    def test_scripts_import_canonical(self):
        for script in ["scripts/train_qlora.py", "scripts/eval_test_set.py",
                       "scripts/verify_adapter.py"]:
            if not Path(script).exists():
                continue
            content = Path(script).read_text(encoding="utf-8")
            assert "from arwen_etl.engine.arwen_prompt import" in content


class TestStakeholderConfig:
    def test_config_exists(self):
        import yaml
        cfg = yaml.safe_load(Path("configs/stakeholders.yaml").read_text(encoding="utf-8"))
        assert len(cfg["stakeholder_groups"]) >= 7

    def test_has_rights_holders(self):
        import yaml
        cfg = yaml.safe_load(Path("configs/stakeholders.yaml").read_text(encoding="utf-8"))
        assert "rights_holder_categories" in cfg
        assert len(cfg["rights_holder_categories"]) >= 10

    def test_has_duty_bearers(self):
        import yaml
        cfg = yaml.safe_load(Path("configs/stakeholders.yaml").read_text(encoding="utf-8"))
        assert "duty_bearer_categories" in cfg
        assert len(cfg["duty_bearer_categories"]) >= 3

    def test_has_hriam_rules(self):
        import yaml
        cfg = yaml.safe_load(Path("configs/stakeholders.yaml").read_text(encoding="utf-8"))
        rules = cfg.get("rules", [])
        assert any("HRIAM" in str(r) or "impact" in str(r).lower() for r in rules)


class TestDatasetStructure:
    @pytest.mark.parametrize("split,expected", [("train", 330), ("validation", 39), ("test", 37)])
    def test_split_counts(self, split, expected):
        path = DATA_DIR / f"{split}.jsonl"
        assert path.exists()
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == expected

    def test_no_refusal_patterns(self):
        for split in ["train", "validation"]:
            for line in Path(DATA_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                for msg in json.loads(line).get("messages", []):
                    if msg.get("role") == "assistant":
                        ans = msg.get("content", "").lower()
                        assert "cannot answer because no source" not in ans

    def test_no_doc_leakage(self):
        def ids(s):
            p = DATA_DIR / f"{s}.jsonl"
            return {d for l in p.read_text(encoding="utf-8").splitlines() if l.strip()
                    for d in json.loads(l).get("source_document_ids", [])}
        assert len(ids("train") & ids("test")) == 0


class TestGenerationPath:
    def test_train_qlora_uses_input_ids(self):
        assert "input_ids=input_ids" in Path("scripts/train_qlora.py").read_text(encoding="utf-8")

    def test_eval_test_set_uses_input_ids(self):
        if Path("scripts/eval_test_set.py").exists():
            assert "input_ids=input_ids" in Path("scripts/eval_test_set.py").read_text(encoding="utf-8")

    def test_verify_adapter_uses_input_ids(self):
        if Path("scripts/verify_adapter.py").exists():
            assert "input_ids=input_ids" in Path("scripts/verify_adapter.py").read_text(encoding="utf-8")

    def test_no_batchencoding_in_generate(self):
        scripts = ["scripts/train_qlora.py", "scripts/eval_test_set.py",
                   "scripts/verify_adapter.py"]
        for s in scripts:
            if not Path(s).exists():
                continue
            c = Path(s).read_text(encoding="utf-8")
            assert "generate(\n                formatted," not in c


class TestTrainingConfig:
    def test_single_gpu_assertion(self):
        c = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "device_count()" in c

    def test_save_total_limit(self):
        c = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "SAVE_TOTAL_LIMIT = 2" in c

    def test_save_only_model(self):
        c = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "save_only_model" in c

    def test_config_validation(self):
        c = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "config_errors" in c

    def test_kaggle_cuda_visible(self):
        c = Path("scripts/kaggle_train.sh").read_text(encoding="utf-8")
        assert "CUDA_VISIBLE_DEVICES=0" in c

    def test_preflight_exists(self):
        assert Path("scripts/preflight.py").exists()


class TestHRIAMGeneration:
    def test_hriam_task_types(self):
        from scripts.build_sft_v2 import HRIAM_TASK_TYPES
        assert len(HRIAM_TASK_TYPES) >= 8

    def test_teacher_has_hriam_guidance(self):
        c = Path("scripts/build_sft_v2.py").read_text(encoding="utf-8")
        assert "HRIAM_NOT_MATERIAL" in c
        assert "HRIAM_RELEVANT" in c
        assert "HRIAM_CENTRAL" in c

    def test_output_format_has_hriam_state(self):
        c = Path("scripts/build_sft_v2.py").read_text(encoding="utf-8")
        fmt_section = c[c.find("Output format"):c.find("Output format") + 800]
        assert "hriam_state" in fmt_section

    def test_parse_carries_hriam_state(self):
        c = Path("scripts/build_sft_v2.py").read_text(encoding="utf-8")
        p_section = c[c.find("def parse_teacher_response"):c.find("def validate_example")]
        assert "hriam_state" in p_section

    def test_validate_allows_policy_no_evidence(self):
        from scripts.build_sft_v2 import validate_example
        ex = {"task_type": "multistakeholder_analysis", "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test question"},
            {"role": "assistant", "content": "test answer"},
        ], "source_document_ids": [], "evidence": []}
        assert validate_example(ex) == []


class TestHRIAMValidator:
    def test_hriam_constants(self):
        from scripts.validate_dataset import HRIAM_TASK_TYPES
        assert len(HRIAM_TASK_TYPES) >= 8

    def test_validator_runs(self):
        from scripts.validate_dataset import validate_dataset
        r = validate_dataset("datasets/sft_final")
        assert "hriam_behavior" in r
        assert "errors" in r


class TestPipeline:
    def test_synthesis_prompt_hriam(self):
        c = Path("src/arwen_engine/pipeline.py").read_text(encoding="utf-8")
        assert "human-rights-aware" in c

    def test_arwen_config(self):
        import yaml
        cfg = yaml.safe_load(Path("configs/arwen.yaml").read_text(encoding="utf-8"))
        assert cfg["deliberation"]["preserve_disagreement"] is True
        assert cfg["deliberation"]["manufacture_consensus"] is False

import os

def test_synthetic_policy_cases_file_exists():
    path = "datasets/benchmark/synthetic_policy_cases.json"
    assert os.path.isfile(path), f"File {path} does not exist"

def test_synthetic_policy_cases_content():
    path = "datasets/benchmark/synthetic_policy_cases.json"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "## Synthetic Benchmark Data Examples" in content
    assert "### Test Case: Ethical AI Policy Framework" in content
    assert "### Test Case: Climate Policy Tensions" in content
    assert '"Document ID":' in content  # though it's not JSON, we expect the pattern
"""Deterministic unit tests for the late-training early stopping callback."""
from __future__ import annotations



def _make_callback():
    """Import and instantiate the LateEarlyStoppingCallback."""
    # The callback is defined inside main() in train_qlora.py. To test it
    # deterministically without importing main(), we replicate the exact
    # callback class here and import the constants from the script.
    import re
    src = open("scripts/train_qlora.py", encoding="utf-8").read()

    # Extract the class body and recreate it
    m = re.search(r"class LateEarlyStoppingCallback\(TrainerCallback\):\n(.*?)\n    class EpochLoggerCallback", src, re.DOTALL)
    assert m, "LateEarlyStoppingCallback class not found in train_qlora.py"
    class_body = m.group(1)

    # Build a minimal namespace to exec the class definition
    ns = {"TrainerCallback": type("TrainerCallback", (), {})}
    exec(f"class LateEarlyStoppingCallback(TrainerCallback):\n{class_body}", ns)
    return ns["LateEarlyStoppingCallback"]


class _FakeControl:
    def __init__(self):
        self.should_training_stop = False


class _FakeState:
    def __init__(self, epoch):
        self.epoch = epoch


class _FakeArgs:
    pass


def _eval(cb, epoch, val_loss):
    state = _FakeState(epoch)
    control = _FakeControl()
    metrics = {"eval_loss": val_loss}
    cb.on_evaluate(_FakeArgs(), state, control, metrics=metrics)
    return control


class TestEarlyStopping:
    """Deterministic tests for late-training early stopping."""

    def test_no_stop_before_epoch_10(self):
        cb = _make_callback()(start_epoch=10, patience=2)
        # Many bad epochs before epoch 10 must NOT stop
        for epoch in range(1, 10):
            control = _eval(cb, epoch, 1.0)  # all non-improving (best is inf)
            assert not control.should_training_stop, f"stopped before epoch 10 at epoch {epoch}"
        assert cb.stopped_epoch is None

    def test_best_metric_established_before_epoch_10(self):
        cb = _make_callback()(start_epoch=10, patience=2)
        # Epoch 1 sets a strong best (low loss)
        _eval(cb, 1, 0.5)
        assert cb.best_metric == 0.5
        # Epochs 2-9 worse — no stop, best stays 0.5
        for epoch in range(2, 10):
            _eval(cb, epoch, 0.9)
            assert cb.best_metric == 0.5
            assert cb.patience_counter == 0  # not yet counting

    def test_epoch_10_improvement_resets_patience(self):
        cb = _make_callback()(start_epoch=10, patience=2)
        _eval(cb, 1, 0.5)  # global best
        # Epoch 10 improves below global best
        control = _eval(cb, 10, 0.4)
        assert not control.should_training_stop
        assert cb.best_metric == 0.4
        assert cb.patience_counter == 0

    def test_epoch_10_non_improvement_increments_patience(self):
        cb = _make_callback()(start_epoch=10, patience=2)
        _eval(cb, 1, 0.5)  # global best
        # Epoch 10 does NOT improve
        control = _eval(cb, 10, 0.6)
        assert not control.should_training_stop
        assert cb.patience_counter == 1

    def test_two_consecutive_late_non_improvements_stop(self):
        cb = _make_callback()(start_epoch=10, patience=2)
        _eval(cb, 1, 0.5)  # global best
        _eval(cb, 10, 0.6)  # non-improve #1
        control = _eval(cb, 11, 0.7)  # non-improve #2
        assert control.should_training_stop
        assert cb.stopped_epoch == 11

    def test_improvement_after_bad_epoch_resets_patience(self):
        cb = _make_callback()(start_epoch=10, patience=2)
        _eval(cb, 1, 0.5)
        _eval(cb, 10, 0.6)  # non-improve #1
        _eval(cb, 11, 0.4)  # improve below 0.5
        assert cb.patience_counter == 0
        # Next two non-improves should stop
        _eval(cb, 12, 0.45)  # non-improve (0.45 > 0.4) #1
        control = _eval(cb, 13, 0.46)  # non-improve #2
        assert control.should_training_stop

    def test_no_more_than_20_epochs(self):
        _make_callback()(start_epoch=10, patience=2)
        # The callback itself never forces more than the configured epochs;
        # it only stops early. This test verifies the script's max epochs.
        src = open("scripts/train_qlora.py", encoding="utf-8").read()
        assert "NUM_EPOCHS = 20" in src

    def test_callback_compares_global_best_not_reset(self):
        cb = _make_callback()(start_epoch=10, patience=2)
        _eval(cb, 1, 0.3)  # very strong global best
        # Epochs 10-11 do not beat 0.3 → patience accumulates
        _eval(cb, 10, 0.4)
        control = _eval(cb, 11, 0.35)
        # 0.35 still worse than 0.3 → non-improve → stop
        assert control.should_training_stop
        # best_metric was NOT reset — still 0.3
        assert cb.best_metric == 0.3


class TestArtifactAndGitSHA:
    def test_git_sha_obtained_via_rev_parse(self):
        src = open("scripts/train_qlora.py", encoding="utf-8").read()
        assert '["git", "rev-parse", "HEAD"]' in src
        assert "REPRODUCIBILITY_FAILURE" in src
        # Must NOT rely on GIT_COMMIT_SHA env var
        assert "GIT_COMMIT_SHA" not in src

    def test_archive_verified_by_opening(self):
        src = open("scripts/train_qlora.py", encoding="utf-8").read()
        assert 'tarfile.open(archive_path, "r:gz")' in src or 'open(archive_path, "r:gz")' in src
        assert "adapter_model.safetensors" in src
        assert "adapter_config.json" in src

    def test_ipc_collect_called(self):
        src = open("scripts/train_qlora.py", encoding="utf-8").read()
        assert "ipc_collect" in src

    def test_artifact_before_qualitative(self):
        # artifact preservation must happen before qualitative eval
        src = open("scripts/train_qlora.py", encoding="utf-8").read()
        artifact_idx = src.find("Artifact preservation")
        qual_idx = src.find("Qualitative evaluation")
        assert artifact_idx != -1 and qual_idx != -1
        assert artifact_idx < qual_idx, "artifact preservation must precede qualitative eval"

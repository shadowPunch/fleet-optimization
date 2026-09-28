import pytest

from dispatch_eval.tracking import tracked_run, tracking_enabled


def test_tracking_disabled_in_tests():
    assert not tracking_enabled()


def test_disabled_run_is_a_noop():
    with tracked_run("t", "test", {"a": 1}) as run:
        run.log({"x": 1.0})
        run.log_table("tbl", [{"a": 1}])
    assert run.name == "untracked"


def test_disabled_run_propagates_exceptions():
    with pytest.raises(RuntimeError), tracked_run("t", "test", {}):
        raise RuntimeError("boom")

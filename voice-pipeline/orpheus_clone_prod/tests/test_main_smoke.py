"""
This is an import-and-wiring smoke test, not a Modal-execution test (Modal
apps are exercised by actually deploying/running them, matching this
repo's established practice of never mocking the training/serving GPU
path -- see the spec's Testing section). It only confirms main.py wires
Tasks 1-5 together without a typo/import error, which is cheap to check
and has caught real mistakes elsewhere in this codebase (e.g. the
"got an unexpected keyword argument" class of bug found by actually
running serve_benchmark.py).
"""
import importlib


def test_main_module_imports_without_error():
    module = importlib.import_module("orpheus_clone_prod.main")
    assert hasattr(module, "app")
    assert hasattr(module, "api")
    assert hasattr(module, "run_training_job_modal")
    assert hasattr(module, "OrpheusCloneEngine")

# Pre-training Diagnostics

Four diagnostic tests that validate each NSCA module works correctly before full training.
Run them from the repository root:

```bash
python diagnostics/run_all.py
```

Or individually:

```bash
python scripts/noisy_tv_test.py
python scripts/forgetting_test.py
python scripts/balloon_test.py
python scripts/slot_discovery_test.py
```

## Tests

| Test | What it checks | Pass condition |
|------|---------------|----------------|
| **Noisy TV** | Curiosity filter rejects unpredictable random signals | Reward on noise < reward on structured transitions |
| **Forgetting** | EWC prevents catastrophic forgetting across tasks | Accuracy on Task 1 stays ≥ 80% after training on Task 2 |
| **Balloon** | Intrinsic reward rises then decays as task is mastered | Reward peaks mid-training, not at the end |
| **Slot Discovery** | Slot attention finds ≥2 distinct object properties | At least 2 slots activate with distinct representations |

## Sample Efficiency Benchmark

The main evaluation is a Meta-World ablation study comparing NSCA vs. a baseline with no physics priors:

```bash
python src/evaluation/metaworld_eval.py
```

Expected result: **≥ 5% accuracy improvement at N=20 samples** (reported: +7.2%).

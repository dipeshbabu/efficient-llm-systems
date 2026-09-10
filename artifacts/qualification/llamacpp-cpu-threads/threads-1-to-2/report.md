Metria verification: VERIFIED

Study: llamacpp-cpu-thread-change
Scope: local llama.cpp CPU thread comparison

Change:
  CPU threads: 1 -> 2

Evidence:
  reference: completed (reference.run.json)
    Observed threads: 1; context: 256
  candidate: completed (candidate.run.json)
    Observed threads: 2; context: 256

Impact:
  kv_fidelity.trajectory_match: completed
    Token prefix agreement: 100/100
    Exact token-sequence matches: 100%
    Divergent prompts: 0
  reference mean process wall time: 0.0842831s (includes startup and model loading)
  candidate mean process wall time: 0.0168433s (includes startup and model loading)

Verdict:
  VERIFIED
  VERIFIED means comparison and analysis completed within the stated scope.
  No task-quality or performance acceptance policy was evaluated.

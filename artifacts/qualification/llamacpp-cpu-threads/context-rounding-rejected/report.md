Metria verification: INSUFFICIENT_EVIDENCE

Study: llamacpp-cpu-thread-change
Scope: local llama.cpp CPU thread comparison

Change:
  CPU threads: 1 -> 2

Evidence:
  reference: completed (reference.run.json)
    Observed threads: 1; context: 256
    runtime-applied context does not match the request
  candidate: completed (candidate.run.json)
    Observed threads: 2; context: 256
    runtime-applied context does not match the request

Impact:
  kv_fidelity.trajectory_match: failed
  reference mean process wall time: 0.0168708s (includes startup and model loading)
  candidate mean process wall time: 0.0187645s (includes startup and model loading)

Verdict:
  INSUFFICIENT_EVIDENCE
  VERIFIED means comparison and analysis completed within the stated scope.
  No task-quality or performance acceptance policy was evaluated.

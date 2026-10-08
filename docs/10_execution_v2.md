# Legacy v2 Series: Execution and Recovery

This document records the v2 execution procedure and historical evidence, not new v3 results.

The protocol applicable to this v2 series is [protocol_jev_qwen4b_v2.md](protocols/protocol_jev_qwen4b_v2.md), including the per-phase reload amendment in Section 8.1. Consult `results/v2/coordinator_reload_state.json` for actual state; this document does not replace execution records.

## Frozen Files

- New test: 100 problems; pilot: 40; development: 20 reused. Construction seed 20261001, inference seed 17. `data/v2/SHA256SUMS` seals the files.
- Applicable configuration: `config/experiment_v2_reload.json`, code 0.2.1, BF16, 40 GB A100, one copy of the Qwen3.5-4B generator and four independent requests per hybrid round.
- Notebooks: `notebooks/v2-reload/`; bundles: `dist/v2-reload/`. Inference without gold; separate analysis with gold.
- `config/experiment_v2.json`, `notebooks/v2/`, `dist/v2/` and the 0.2.0 results remain historical evidence. They are neither regenerated nor resumed under the new code.
- Human review omitted with user authorization: `reviewed=false`. Do not present the data as reviewed.

## Continuity

Each VM has its own CLI configuration, executable wrapper, durable operator and guard. The guard maintains WebSocket every 15 s and checks the assignment every 20 s. The operator polls every 25 s, downloads checkpoints during the run, performs progress checks and limits the VM to four hours.

A transport error does not prove that the notebook has died. Guard recovery is bounded to 180 s; it does not restart the kernel or inference. Shutdown is considered complete only after confirming the endpoint's absence on the server.

The coordinator runs in a WSL process independent of the agent's terminal, with a global limit of 12 hours and at most two assignments. Other assignments consume capacity but are never stopped. A WSL host process prevents premature shutdown of its children during startup.

Do not close WSL, suspend the computer or modify code, data or bundles while work is active. Pings do not guarantee immunity to service revocation, network loss or local shutdown.

## Sequence

1. Revalidate historical development evidence: three hybrids, 4B alone and the 0.8B diagnostic on A100. Do not repeat those inferences.
2. Repeat only the failed minimal latency run: 2 development questions and 4 synthetic ones, 36 measurements. Fully shut down B before loading G4+J, and vice versa. Loading, probes and rewarming are outside `T_total`.
3. Pilot: selection among complete solutions and the large model with sampling, 40 cases each; then 4B alone, 40 cases.
4. Apply `tools/pilot_decision.py` with the applicable configuration. No-go stops the experiment. It does not authorize changes to prompts, seeds, weights or budgets.
5. Only with go: confirmatory runs in pairs [large model with sampling, large model greedy], [4B alone, block selection], [stepwise selection, final selection]. One hundred cases per arm.
6. Full latency: one A100, 192 rows, with reloads outside the timer. CPU analysis with exactly seven final ZIPs; independent audit.

## Check and Recover

```powershell
# Read-only: actual server assignments.
wsl -e colab sessions
```

First consult `coordinator_reload_state.json`, `coordinator_reload_events.jsonl` and the job's `status.json`. State distinguishes execution, verification and release. Each job preserves `execution_handover.json`, logs, checkpoints, final ZIPs and the executed notebook.

The following command only inspects preparation; it does not assign VMs:

```bash
python3 tools/colab/run_v2.py --plan --amend-latency-reload
```

Do not launch a second coordinator while the existing one is alive. Recovery uses `--run --amend-latency-reload`, the same state files and their ownership proofs; it neither replaces failed jobs nor repeats inference to improve results. If an ambiguous handover appears, inspect the process and assignment before deciding on any action.

## Preserved Incidents

- `results/v2/preparation/`: initial candidate rejected because of an incomplete monetary filter, and a candidate rejected because of an unauthorized seed change. The final selection uses the original seed.
- `results/v2/smoke/hybrid/`: three notebooks completed, verified and released. Observed stress of 36.12 GiB, J input of 16,330 tokens; graph/eager agreement 1.0 on three initial prompts.
- `results/v2/smoke/single_lat/`: 4B alone completed and verified by `gsingle_verification.json`; the combined operator remains marked as failed because of the subsequent latency OOM. Successful completion of the sequence has not been fabricated.
- `results/v2/smoke/diagnostic_08b/`: development diagnostic completed; not used for quality conclusions.
- The latency OOM was not a disconnection: sleeping B retained 2.83 GiB and G could not reserve its 3 GiB of KV. The new configuration keeps the same caches and changes only residency management between phases.

Reload times are published separately. Measured latency is conditional on loaded, warm engines, not the end-to-end cost of swapping models. CU and monetary costs are not invented: operational estimates explicitly identify when only observed time is available.

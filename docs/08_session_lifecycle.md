# Colab Supervision and Shutdown

This document records the legacy lifecycle procedure and its 2026-09-29 verification, not new Qwen4B v3 results.

The lifecycle is independent of experimental monitoring at 2/5/10/30 minutes.

- WebSocket pings: every 15 seconds, connecting ONLY to the existing CLI kernel.
- Server assignment check: every 20 seconds, with a network timeout.
- Connection renewal with updated proxy credentials: before 50 minutes.
- Guard cancellation: checked every second while waiting for the next check.
- Shutdown: unassign the exact assignment, then list_assignments to confirm its
  absence. Up to three bounded attempts; a failed network response does NOT prove shutdown.
- Log: results/lifecycle/<SESSION>/session_lifecycle.jsonl, without tokens or credentials.

## Usage

The launch.sh and launch_seq.sh scripts start the guard before uploading/running the notebook.
stop.sh releases the VM and checks the server. If a new VM fails during launch,
a trap attempts to release it; if it cannot confirm shutdown, it returns an error and a warning.

For tools that isolate CLI state, export COLAB_SESSION_CONFIG with the same
path used by your wrapper executable. The default is ~/.config/colab-cli/sessions.json.
COLAB_PYTHON specifies the Python from the environment where the CLI is installed.

No spare kernels are created and no dummy cells are executed to maintain activity.
The guard waits until launch has created the CLI kernel, checks the
assignment and maintains its connection. It does not guarantee availability or bypass Colab
quotas or limits. A failed ping/observation is logged as an unknown state.

## Actual Verification

Integration test on 2026-09-29 on a temporary CPU VM, without models:

- WebSocket connection confirmed and five health observations spaced 20 seconds apart.
- Pings configured at 15 seconds in the installed WebSocket client.
- Cancellation at 22:34:13 UTC; guard stopped at 22:34:14 UTC.
- Assignment absence confirmed on the first attempt, at 22:34:14 UTC.
- Guard process absent and global session list empty after shutdown.
- Evidence: results/first_two_20260929/guard_check/session_lifecycle.jsonl.

The first attempt to use the HTTP TFE keep-alive endpoint returned HTTP400 on this account;
that implementation was discarded and the CLI WebSocket connection was verified. That attempt
is not presented as a successful keep-alive.

The G_SINGLE/B13 experiments had already finished and their VMs had been released before
the new guard's integration test; their results were neither changed nor repeated.

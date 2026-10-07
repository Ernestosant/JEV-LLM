"""Exception-safe v3 adapter; the shared request/termination loop stays unchanged."""

from __future__ import annotations

import threading
import time
import traceback

from ..engines import Engine


class SafeEngine(Engine):
    def __init__(self, *args, cleanup_timeout_s=5.0, cleanup_max_steps=64,
                 failure_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.usable = True
        self.cleanup_timeout_s = float(cleanup_timeout_s)
        self.cleanup_max_steps = int(cleanup_max_steps)
        self.failure_callback = failure_callback
        self.last_cleanup = None

    def _cleanup_requests(self, engine, attempted, *, only_if_unfinished=False):
        started = time.perf_counter()
        report = {"attempted_ids": list(attempted), "cleanup_confirmed": False,
                  "unfinished": None, "steps": 0, "timeout_s": self.cleanup_timeout_s}
        completed = threading.Event()

        def cleanup():
            try:
                if only_if_unfinished and engine.has_unfinished_requests() is False:
                    report.update(unfinished=False, cleanup_confirmed=True)
                    return
                # Include the ID whose add_request accepted work and then raised.
                engine.abort_request(list(attempted))
                for _ in range(self.cleanup_max_steps + 1):
                    unfinished = engine.has_unfinished_requests()
                    report["unfinished"] = unfinished if type(unfinished) is bool else None
                    if unfinished is False:
                        report["cleanup_confirmed"] = True
                        break
                    if report["steps"] >= self.cleanup_max_steps or time.perf_counter() - started >= self.cleanup_timeout_s:
                        break
                    engine.step()
                    report["steps"] += 1
            except Exception:
                report["cleanup_traceback"] = traceback.format_exc()
            finally:
                completed.set()

        # A broken native/RPC call must not prevent the parent from stopping this engine.
        threading.Thread(target=cleanup, daemon=True).start()
        finished = completed.wait(max(0.0, self.cleanup_timeout_s))
        result = dict(report)
        result["cleanup_confirmed"] = finished and report["cleanup_confirmed"] is True
        result["seconds"] = time.perf_counter() - started
        result["timed_out"] = not finished
        self.last_cleanup = result
        if not result["cleanup_confirmed"]:
            self.usable = False
        return result

    def run(self, reqs, deadline=None):
        if not self.usable:
            exc = RuntimeError(f"engine {self.name} is unusable after unconfirmed request cleanup")
            exc.engine_cleanup = self.last_cleanup
            raise exc
        engine, attempted, owner = self.engine, [], self

        class TrackedEngine:
            def __getattr__(self, name):
                return getattr(engine, name)

            def add_request(self, rid, *args, **kwargs):
                attempted.append(rid)
                return engine.add_request(rid, *args, **kwargs)

            def abort_request(self, ids):
                completed, errors = threading.Event(), []

                def abort():
                    try:
                        engine.abort_request(ids)
                    except Exception as exc:
                        errors.append(exc)
                    finally:
                        completed.set()

                started = time.perf_counter()
                threading.Thread(target=abort, daemon=True).start()
                if not completed.wait(max(0.0, owner.cleanup_timeout_s)):
                    owner.usable = False
                    owner.last_cleanup = {"attempted_ids": list(attempted), "abort_ids": list(ids),
                                          "cleanup_confirmed": False, "unfinished": None, "steps": 0,
                                          "timeout_s": owner.cleanup_timeout_s, "timed_out": True,
                                          "seconds": time.perf_counter() - started}
                    exc = TimeoutError("request termination abort did not return within cleanup bound")
                    exc.engine_cleanup = owner.last_cleanup
                    raise exc
                if errors:
                    raise errors[0]

        self.engine = TrackedEngine()
        try:
            results = super().run(reqs, deadline)
            if any(r.finish_reason in ("timeout", "final_abort", "abort") for r in results.values()):
                cleanup = self._cleanup_requests(engine, attempted, only_if_unfinished=True)
                if not cleanup["cleanup_confirmed"]:
                    exc = RuntimeError("request termination cleanup could not be confirmed")
                    exc.engine_cleanup = cleanup
                    raise exc
            return results
        except Exception as exc:
            failure = {"kind": "engine_exception", "engine_role": self.name,
                       "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc(), "attempted_ids": list(attempted)}
            # Persist the original failure before attempting cleanup, which can also fail.
            try:
                if self.failure_callback:
                    self.failure_callback({**failure, "cleanup_pending": True})
            finally:
                if not hasattr(exc, "engine_cleanup"):
                    exc.engine_cleanup = self._cleanup_requests(engine, attempted)
            if self.failure_callback:
                self.failure_callback({**failure, "cleanup_pending": False,
                                       "engine_cleanup": exc.engine_cleanup, "engine_usable": self.usable})
            raise
        finally:
            self.engine = engine

"""jevlab: v3/v3.1 and isolated 4B v2 experimental series on vLLM.

Modules are split so that the offline parts (parsing, seeds, evaluate, analysis) import
nothing heavy; GPU parts (engines, selector, algorithms, runner) import vLLM lazily.
"""

# Bump on every change that can alter what is measured: it is part of the config_hash, so a
# repaired pipeline never resumes (or mixes with) runs produced by the previous code.
# 0.1.1: EOS id stripped from delivered token_ids (was leaking '<|im_end|>' into outputs).
# 0.1.2: notebook 08 synthetic prompts redesigned so every band fits JSTEP's 64-round cap.
# 0.2.0: isolated 4B series, split/config routing and stricter preflight validation.
# 0.2.1: approved same-GPU latency unload/reload amendment after sleep-mode OOM.
__version__ = "0.2.1"

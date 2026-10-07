"""Native JevK5 reference (subprocess): python -m jevlab.native_ref in.json out.json snapshot

Loads the published runtime `jevk5.JevK5(snapshot, graphs=False)` (transformers path) and
returns, for each (state, question), the calibrated probabilities, raw letter logits, the
input token ids hash and length. Run BEFORE any vLLM engine is created and in its own
process, so its CUDA context never pollutes vLLM's memory check.
"""

import hashlib
import json
import sys
import time


def main(inp: str, out: str, snapshot: str) -> None:
    import torch
    from jevk5 import JevK5
    from jevk5.prompt import decision_options, prompt_text

    items = json.load(open(inp))
    t0 = time.perf_counter()
    model = JevK5(snapshot, graphs=False)
    load_s = time.perf_counter() - t0
    res = {"load_s": load_s, "temperature": model.temperature, "items": [],
           "torch": torch.__version__}
    for it in items:
        state, q = it["state"], it["question"]
        options = decision_options(q)
        texts = [t for _, t in options]
        ids = model.encode(state, q["instructions"], texts)
        ids_pt = model.tok.encode(prompt_text(state, q["instructions"], texts), add_special_tokens=False)
        t1 = time.perf_counter()
        logits = model.letter_logits(ids, len(texts)).tolist()
        probs, tokens = model.probabilities(state, q)
        res["items"].append({
            "n_ids": len(ids), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "ids_equal_prompt_text": ids == ids_pt, "raw_logits": logits,
            "probabilities": [probs[k] for k, _ in options], "input_tokens": tokens,
            "seconds": time.perf_counter() - t1})
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

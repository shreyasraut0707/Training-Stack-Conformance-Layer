"""Is P1's update gap noise or a systematic scale error? (docs/m0_findings.md)

For d = g_trainer - g_ref, a cosine(d, g_ref) near +/-1 with a matching norm error
means the update is scaled wrong; a cosine near 0 that changes sign across seeds
means the difference is spread like noise. Runs 3 seeds x 2 models x 2 trainers.
"""
import copy

import torch

from conform.noise_floor import reference_step
from conform.probes.p1_grad_accum import build_rows, run_trainer
from conform.proxy_model import DEFAULT_TARGETS, build_proxy

for model_id in DEFAULT_TARGETS:
    for seed in (0, 1, 2):
        model, tokenizer, _ = build_proxy(model_id, seed=seed)
        pad_id = next(i for i in (tokenizer.pad_token_id, tokenizer.eos_token_id, 0) if i is not None)
        rows = build_rows(len(tokenizer), pad_id, seed=seed)
        micro_batches = [{k: v.unsqueeze(0) for k, v in row.items()} for row in rows]
        _, ref = reference_step(model, micro_batches)
        for trainer in ("transformers", "trl"):
            _, grad = run_trainer(copy.deepcopy(model), tokenizer, rows, trainer=trainer,
                                  batch_size=1, accumulation=4, seed=seed, trainer_cls=None)
            diff = grad - ref
            print(f"{model_id} seed={seed} {trainer:12} "
                  f"gap={(diff.norm() / ref.norm()).item():.2e}  "
                  f"cos={torch.nn.functional.cosine_similarity(diff, ref, dim=0).item():+.3f}  "
                  f"norm_error={(grad.norm() / ref.norm()).item() - 1:+.2e}")

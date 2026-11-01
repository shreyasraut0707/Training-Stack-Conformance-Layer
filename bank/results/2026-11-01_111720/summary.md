| Case | Environment | Model | transformers | trl | torch | Status | Expected | Outcome |
|---|---|---|---|---|---|---|---|---|
| eos-masked-pad-eq-eos-2025-behaviour | defective | HuggingFaceTB/SmolLM2-135M-Instruct | 4.50.3 | 0.16.0 | 2.6.0 | fail | fail | match |
| eos-masked-pad-eq-eos-2025-behaviour | fixed | HuggingFaceTB/SmolLM2-135M-Instruct | 4.50.3 | 0.16.1 | 2.6.0 | pass | not_fail | match |

| Case | Environment | Model | transformers | trl | torch | Status | Expected | Outcome |
|---|---|---|---|---|---|---|---|---|
| eos-masked-pad-eq-eos-2025 | defective | HuggingFaceTB/SmolLM2-135M-Instruct | 4.50.3 | 0.16.0 | 2.6.0 | fail | fail | match |
| eos-masked-pad-eq-eos-2025 | fixed | HuggingFaceTB/SmolLM2-135M-Instruct | 4.50.3 | 0.16.1 | 2.6.0 | pass | not_fail | match |
| ga-normalisation-2024 | defective | Qwen/Qwen2.5-0.5B-Instruct | 4.45.2 | – | 2.4.1 | fail | fail | match |
| ga-normalisation-2024 | defective | HuggingFaceTB/SmolLM2-135M-Instruct | 4.45.2 | – | 2.4.1 | fail | fail | match |
| ga-normalisation-2024 | fixed | Qwen/Qwen2.5-0.5B-Instruct | 4.46.3 | – | 2.4.1 | fail | not_fail | mismatch |
| ga-normalisation-2024 | fixed | HuggingFaceTB/SmolLM2-135M-Instruct | 4.46.3 | – | 2.4.1 | fail | not_fail | mismatch |

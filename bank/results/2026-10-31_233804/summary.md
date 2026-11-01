| Case | Environment | Model | transformers | trl | torch | Status | Expected | Outcome |
|---|---|---|---|---|---|---|---|---|
| eos-masked-pad-eq-eos-2025 | defective | HuggingFaceTB/SmolLM2-135M-Instruct | – | – | – | not_tested | fail | not tested |
| eos-masked-pad-eq-eos-2025 | fixed | HuggingFaceTB/SmolLM2-135M-Instruct | – | – | – | not_tested | not_fail | not tested |
| ga-normalisation-2024 | defective | Qwen/Qwen2.5-0.5B-Instruct | – | – | – | not_tested | fail | not tested |
| ga-normalisation-2024 | defective | HuggingFaceTB/SmolLM2-135M-Instruct | – | – | – | not_tested | fail | not tested |
| ga-normalisation-2024 | fixed | Qwen/Qwen2.5-0.5B-Instruct | – | – | – | not_tested | not_fail | not tested |
| ga-normalisation-2024 | fixed | HuggingFaceTB/SmolLM2-135M-Instruct | – | – | – | not_tested | not_fail | not tested |

Reasons:
- eos-masked-pad-eq-eos-2025 / defective / HuggingFaceTB/SmolLM2-135M-Instruct: environment bank-eos-defective not created
- eos-masked-pad-eq-eos-2025 / fixed / HuggingFaceTB/SmolLM2-135M-Instruct: environment bank-eos-fixed not created
- ga-normalisation-2024 / defective / Qwen/Qwen2.5-0.5B-Instruct: environment bank-ga-defective not created
- ga-normalisation-2024 / defective / HuggingFaceTB/SmolLM2-135M-Instruct: environment bank-ga-defective not created
- ga-normalisation-2024 / fixed / Qwen/Qwen2.5-0.5B-Instruct: environment bank-ga-fixed not created
- ga-normalisation-2024 / fixed / HuggingFaceTB/SmolLM2-135M-Instruct: environment bank-ga-fixed not created

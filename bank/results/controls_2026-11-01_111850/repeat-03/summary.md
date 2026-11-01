| Case | Environment | Model | transformers | trl | torch | Status | Expected | Outcome |
|---|---|---|---|---|---|---|---|---|
| controls | p4-trl | Qwen/Qwen2.5-0.5B-Instruct | 5.22.1 | 1.17.1 | 2.15.0+cpu | pass | not_fail | match |
| controls | p4-trl | HuggingFaceTB/SmolLM2-135M-Instruct | 5.22.1 | 1.17.1 | 2.15.0+cpu | pass | not_fail | match |
| controls | p4-trl | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 5.22.1 | 1.17.1 | 2.15.0+cpu | pass | not_fail | match |
| controls | p5-transformers | Qwen/Qwen2.5-0.5B-Instruct | 5.22.1 | 1.17.1 | 2.15.0+cpu | pass | not_fail | match |
| controls | p5-transformers | HuggingFaceTB/SmolLM2-135M-Instruct | 5.22.1 | 1.17.1 | 2.15.0+cpu | pass | not_fail | match |
| controls | p5-transformers | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 5.22.1 | 1.17.1 | 2.15.0+cpu | pass | not_fail | match |
| controls | p5-trl | Qwen/Qwen2.5-0.5B-Instruct | – | – | – | not_tested | not_fail | not tested |
| controls | p5-trl | HuggingFaceTB/SmolLM2-135M-Instruct | – | – | – | not_tested | not_fail | not tested |
| controls | p5-trl | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | – | – | – | not_tested | not_fail | not tested |
| controls | p1m-transformers | Qwen/Qwen2.5-0.5B-Instruct | – | – | – | not_tested | not_fail | not tested |
| controls | p1m-transformers | HuggingFaceTB/SmolLM2-135M-Instruct | – | – | – | not_tested | not_fail | not tested |
| controls | p1m-transformers | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | – | – | – | not_tested | not_fail | not tested |

Reasons:
- controls / p5-trl / Qwen/Qwen2.5-0.5B-Instruct: probe crashed (exit 1073807364): no error output
- controls / p5-trl / HuggingFaceTB/SmolLM2-135M-Instruct: probe crashed (exit 3221226091): no error output
- controls / p5-trl / TinyLlama/TinyLlama-1.1B-Chat-v1.0: probe crashed (exit 3221226091): no error output
- controls / p1m-transformers / Qwen/Qwen2.5-0.5B-Instruct: probe crashed (exit 3221226091): no error output
- controls / p1m-transformers / HuggingFaceTB/SmolLM2-135M-Instruct: probe crashed (exit 3221226091): no error output
- controls / p1m-transformers / TinyLlama/TinyLlama-1.1B-Chat-v1.0: probe crashed (exit 3221226091): no error output

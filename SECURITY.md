# Security and official sources

## Official sources only

This tool runs training code and can be registered into coding agents, which makes it an
attractive target for impersonation (spec Section 12.2). Install it **only** from:

- the official repository: *to be announced when published*
- the official PyPI project: *to be announced when published* (`conform` is a working
  name only: that name belongs to an unrelated PyPI project, so **any `conform` package on
  PyPI is not this tool**)

Official releases are built by `.github/workflows/release.yml` and published with PyPI
Trusted Publishing and provenance attestations. A copy that does not come from these
sources, or a release without attestations, is not official.

## What the tool does and does not do

- **Local only.** No network calls except the config and tokenizer downloads you request
  (`--download`); never model weights. No rows or reports are uploaded. No telemetry.
- **Never asks for secrets**, including Hugging Face tokens.
- **Executes the installed training code** and loads tokenizers: run it in a throwaway
  environment when the recipe or rows are untrusted. `trust_remote_code` stays off.
- Each probe runs in its own time-limited process with a temporary directory that is
  removed afterwards.
- The MCP server uses the local stdio transport only.
- `conform setup` only shows its planned changes unless you pass `--apply`, and backs up
  every file it changes.

## Reporting a vulnerability

Until an official repository exists, report privately to the project owner rather than in
a public issue.

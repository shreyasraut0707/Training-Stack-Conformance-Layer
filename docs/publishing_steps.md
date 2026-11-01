# Repository, CI and publishing: step by step

Nothing has been published. Everything public (a public repository, a PyPI upload) waits
for your explicit approval. Steps marked **(you)** need your accounts; the rest is done
from this computer once you approve.

## 1. Choose the final name (spec Section 14.1) (you)

Checked on 2026-11-01: `conform` is **taken** on PyPI (an unrelated 2019 project) and as a
GitHub user; npm could not be checked. `training-stack-conform` and `conform-probes` are
free on PyPI (free is not reserved: whoever uploads first gets a name). Choose a name and
tell me; the package name in `pyproject.toml`, the docs and `SECURITY.md` are then updated
together and the tests rerun. Until then **no upload is possible**: PyPI would refuse
`conform`.

## 2. GitHub repository and CI

The repository is **private** (your choice, 2026-11-01) and does not contain the spec PDF
(`.gitignore` leaves out `*.pdf`, environments, caches, build output and generated
reports). `bank/results/` is included on purpose: it is the evidence behind
`docs/bank_results.md`. Status of the repository and of the first CI run:
`docs/PROGRESS.md`.

**What CI runs** (`.github/workflows/ci.yml`): on every push, the tests and the G3 controls
on the pinned stack (must pass) and on the latest releases (informational), and the
regression bank on Windows; every Monday and on "Run workflow", a check against upstream
`main` of Transformers and TRL. **Pass if:** the `tests (pinned)` job is green and the
`bank` job's summary shows every row `match`.

**Cost of a private repository.** On GitHub's Free plan, Actions runs in private
repositories draw on an included monthly amount (about USD 12 of runner time; Windows
minutes cost more than Linux minutes). When it is used up, runs stop; nothing is charged
unless you add a payment method and a budget. One full CI run takes roughly an hour of
Linux time plus up to two hours of Windows time, so expect on the order of ten full runs
a month. To limit waste, a newer push cancels an older run of the same branch and every
job has a time limit. Public repositories do not use this allowance.

To view runs: the repository's **Actions** tab, or in a terminal:

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" run list --limit 5
```

## 3. Publish to PyPI with Trusted Publishing (spec Section 12.2)

### What Trusted Publishing is (no passwords, no tokens)

Normally an upload to PyPI needs a password or an API token, and a token stored in GitHub
could leak. With **Trusted Publishing**, you tell PyPI once: "accept uploads of my project
that come from **this** GitHub repository's **release.yml** workflow". When that workflow
runs, GitHub gives it a short-lived, signed identity token (OpenID Connect); PyPI checks
that it matches what you registered and accepts the upload. Nothing secret is stored
anywhere. **Attestations** are signed records, published next to each file on PyPI, that
say which repository and workflow built it; `SECURITY.md` tells users to trust only
attested releases.

A **pending publisher** is that registration made **before** the project exists on PyPI.
The first successful upload creates the project. It does **not** reserve the name: if
someone else uploads that name first, the pending publisher stops working.

### What is already prepared

- `.github/workflows/release.yml`: builds the package, checks it with `twine check
  --strict`, and then either
  - **rehearses** on TestPyPI (PyPI's separate practice site) when you press
    "Run workflow" by hand, using the GitHub environment `testpypi`; or
  - **publishes** to the real PyPI when you publish a GitHub release, using the
    environment `pypi`, with attestations.
- The package builds without warnings and passes `twine check --strict` (checked on this
  computer, 2026-11-01). It contains the code, the skill, `README.md` and `LICENSE`; not
  the spec PDF, the results or the tests' data.

### Recommended order

**Step A: create the two accounts (you, about 10 minutes, free).**

1. Go to <https://pypi.org/account/register/>, register with your e-mail address, and
   click the link in the confirmation e-mail.
2. PyPI requires **two-factor authentication**: under *Account settings*, add an
   authenticator app (for example Microsoft Authenticator or Google Authenticator) or a
   security key. Save the **recovery codes** somewhere safe and offline.
3. Do the same on TestPyPI: <https://test.pypi.org/account/register/> (a separate site
   with a separate account).

Never send me a password, a recovery code or a token; I never need them.

**Step B: decide when the repository becomes public (you).** Recommended: make it public
just before the first real release, after contacting the TRL and Transformers maintainers
(spec Section 14). Reasons: a PyPI package is public anyway; on the Free plan GitHub
**environments with a required approval** only work in public repositories, and that
approval is what makes every real upload wait for your click; the attestations name the
repository. Making it public is an outward-facing change: I will do it only when you say so.

**Step C: tell me the final name** (section 1). I update the files, rerun the tests and
push.

**Step D: register the pending publishers (you, on the two websites).**

1. On TestPyPI: *Your account* → *Publishing* → *Add a new pending publisher* → tab
   **GitHub**, and fill in:

   | Field | Value |
   |---|---|
   | PyPI Project Name | the final name |
   | Owner | your GitHub user name |
   | Repository name | `Training-Stack-Conformance-Layer` |
   | Workflow name | `release.yml` |
   | Environment name | `testpypi` |

2. On PyPI (<https://pypi.org/manage/account/publishing/>): the same, with Environment
   name **`pypi`**.

**Step E: create the two GitHub environments (you, once the repository is public).**
Repository → *Settings* → *Environments* → *New environment*:

1. `testpypi`: no extra settings.
2. `pypi`: tick **Required reviewers**, add yourself, *Save protection rules*. From then
   on, every real upload waits until you click *Approve* on the run.

**Step F: rehearse on TestPyPI.** Repository → *Actions* → *release* → *Run workflow*
(or ask me to start it). Then check <https://test.pypi.org/project/NAME/> and try an
install in a throwaway environment:

```powershell
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ NAME
```

**Step G: the real release (only with your explicit go-ahead).** The version in
`pyproject.toml` and `src/conform/__init__.py` is set to the release number (for example
`0.2.0`), the change is pushed, and a GitHub release with the tag `v0.2.0` is published.
`release.yml` builds the package, waits for your approval in the `pypi` environment, and
uploads it with attestations. **An upload cannot be undone or replaced:** a version
number can be used only once on PyPI (a bad release can be "yanked", not overwritten).

Last, fill in the official repository and PyPI links in `SECURITY.md`.

## 4. Optional, recommended by the spec

- Contact the TRL and Transformers maintainers before any public claim (spec Section 14).
- Offer the bank cases upstream as regression tests (spec Section 12.3).
- A small user pilot: fine-tuners run `conform check` on real recipes and record whether any
  finding was new to them (spec Section 10.2, step 6).

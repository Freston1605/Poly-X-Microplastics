# Repository settings

The protections that do not live in YAML. Merge protection, tag protection and
the security switches are repository state: a fork reset, a settings migration
or a new maintainer loses them without a red build, because no workflow can
assert them. This file is the record of what is switched on, why, and how to
put it back.

`docs/adr/` records decisions; this file records state. When a setting
changes, change it here in the same pull request.

## Order matters

Apply these **after** the workflow change that names the jobs they refer to
has landed on `main`, and apply the required-checks list *last*. GitHub
matches a required check by the name a job reports: a context that no job
produces leaves every pull request waiting forever, which blocks the merge
that would have fixed it.

## The switches

| Switch | Value | Where |
| --- | --- | --- |
| Ruleset `Protect release tags` | `creation`, `update` and `deletion` of `refs/tags/v*`, bypass for admin and maintain | Repository → Rules → Rulesets |
| Require actions to be pinned to a full-length commit SHA | on | Settings → Actions → General |
| Private vulnerability reporting | on | Settings → Code security |
| Dependabot alerts | on | Settings → Code security |
| Dependabot security updates | on | Settings → Code security |
| Required status checks | the six checks of the merge contract (below), strict up-to-date on | Default-branch ruleset |

Every call below needs a token with **admin** on
`Freston1605/Poly-X-Microplastics`. A 403 means the token lacks admin. Read
every setting back before concluding anything: `vulnerability-alerts` also
answers 404 when the alerts are off, so a 404 is not always a permission
problem.

## Release tags are maintainer-only

`publish.yml` publishes to PyPI on any `v*` tag. Without a ruleset the tag is
a public button: anyone who can push a branch can push a tag and spend a
version number that cannot be reused. The ruleset makes the button
maintainer-only.

| Field | Value |
| --- | --- |
| Name | `Protect release tags` |
| Target | `tag` |
| Enforcement | `active` |
| Conditions | `ref_name.include = ["refs/tags/v*"]` |
| Rules | `creation`, `update`, `deletion` |
| Bypass | `RepositoryRole` 2 (maintain), 5 (admin), mode `always` |

GitHub's model is inverted with respect to the wording: a rule applies to
**everyone except** the bypass list. So "only maintainers can create a tag"
is the `creation` rule plus an admin/maintain bypass — not "allow maintainers"
plus an empty list, which would let anyone tag.

`update` is not decoration. Without it a tag can be force-moved to another
commit after the fact, so the release record stops matching the code it names;
with it, the only way to retarget a release is to delete the tag and create it
again, which the same ruleset refuses to non-maintainers. `creation` and
`deletion` alone leave that door open.

```bash
gh api repos/Freston1605/Poly-X-Microplastics/rulesets \
  --jq '.[] | {id, name, target, enforcement}'
```

The role IDs are **not** the order of the roles on the settings page: maintain
is `2`, write is `4`, admin is `5`. A guessed `4` reads as "maintain" and in
fact hands the bypass to everyone with write access — the opposite of the
rule. Both IDs are checked against the ruleset's own `bypass_actors` after
creation; the ruleset *list* omits that field, so only the detail endpoint
shows who can bypass.

To create it on a fresh repository (as applied 2026-11-15, ruleset id
`27341653`; check the list above first — a second identical ruleset adds
nothing and is one more thing to keep in step):

```bash
gh api --method POST \
  repos/Freston1605/Poly-X-Microplastics/rulesets \
  --input - <<'JSON'
{
  "name": "Protect release tags",
  "target": "tag",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["refs/tags/v*"], "exclude": []}},
  "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}],
  "bypass_actors": [
    {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"},
    {"actor_id": 2, "actor_type": "RepositoryRole", "bypass_mode": "always"}
  ]
}
JSON
```

Tags that do not start with `v` are untouched: a scratch tag is still cheap.
The ruleset has nothing to do with the build; it guards the *trigger*.

## Action refs must be full commit SHAs

Every `uses:` in `.github/workflows/` is already pinned to a 40-character SHA
with its release in a comment, and `.github/dependabot.yml` keeps those pins
current. That was a convention, and a convention fails silently: a new action
added as `owner/action@v1` runs the tag, and a moved tag runs different code
than the review approved.

With the platform setting on, a workflow whose action ref is not a full SHA
fails at run start, before the job does any work. The reviewer is no longer
the enforcement. The setting covers **actions**; a reusable workflow is still
allowed by tag, which is why this repository calls its own gate by path
(`uses: ./.github/workflows/ci.yml`) and has no remote reusable workflow to
police.

```bash
gh api repos/Freston1605/Poly-X-Microplastics/actions/permissions
# {"enabled":true,"allowed_actions":"all","sha_pinning_required":true}
```

`PUT` replaces the object, so send all three fields even when only one
changes. Read it first if the repository has custom `allowed_actions`:

```bash
gh api --method PUT repos/Freston1605/Poly-X-Microplastics/actions/permissions \
  -F enabled=true -f allowed_actions=all -F sha_pinning_required=true
```

The response is `204 No Content`; the `GET` above is the evidence.

## The reporting channel and the advisory feed

`SECURITY.md` asks reporters not to open a public issue. Private vulnerability
reporting is the private channel behind that sentence: the report arrives as
a repository advisory, stays private until disclosure, and is the route
through which GitHub can issue a CVE for it. With it off, the policy promises
something the repository cannot accept.

```bash
gh api --method PUT \
  repos/Freston1605/Poly-X-Microplastics/private-vulnerability-reporting \
  -F enabled=true
gh api repos/Freston1605/Poly-X-Microplastics/private-vulnerability-reporting
# {"enabled":true}
```

Dependabot alerts and security updates are the other half. Every dependency
here is pinned exactly on purpose, so the advisory feed is the only thing that
turns "this pin is now known-vulnerable" into a version somebody moves. The
alerts show the affected package in the advisory; the security updates open a
pull request that raises the pin toward the first fixed release. Because the
pins live in two places — `pyproject.toml` and the frozen `requirements.txt` —
treat that pull request as a starting point, not a merge-ready change: both
manifests have to move together, and `uv lock` has to follow.

```bash
gh api --method PUT repos/Freston1605/Poly-X-Microplastics/vulnerability-alerts
gh api --method PUT repos/Freston1605/Poly-X-Microplastics/automated-security-fixes
gh api repos/Freston1605/Poly-X-Microplastics/automated-security-fixes
# {"enabled":true,"paused":false}
```

Both `PUT`s answer `204 No Content`. `vulnerability-alerts` answers `204` with
an empty body, and `404` when the alerts are off: that status code is the
verification.

### Security updates are only on when the uv pin is satisfiable

Turning "Dependabot security updates" on is half a switch here. For the Python
ecosystem the other half is the uv version in `pyproject.toml`, because
Dependabot does not install it: its uv updater bundles exactly one uv and runs
`uv lock` with that binary. `required-version` is read first, and a value it
does not satisfy ends the job before any lock work:

```
| Dependency | Error Type                 | Error Details                       |
| torch      | tool_version_not_supported | { "tool-name": "uv",                |
|            |                            |   "detected-version": "==0.11.9",   |
|            |                            |   "supported-versions": "0.13.7" }  |
```

That is run `43796115788` (2026-11-16, `uv` in `/`, torch; the Pillow job
`43796112000` failed the same way). No pull request is opened; the only
trace is a red run whose name says "Dependabot Updates failure". The setting
looks on and the Python half is silent. #107 and #110 exist because of this.

The rule, therefore:

- `required-version` in `pyproject.toml` must be **exactly** the uv that
  Dependabot bundles, and it is the single source of truth: no workflow passes
  a `version:` to `setup-uv`, so CI installs the same binary that writes the
  lockfile. An exact pin is deliberate — a range would let two resolvers write
  and read one lock.
- When Dependabot's bundled uv moves, the failure above names both versions.
  The fix is a planned change, not a mystery: bump `required-version`, run
  `uv lock` with that same binary, commit the pin and the lock together, and
  re-sync local environments (`uv sync --locked --extra gui`) because the old
  `.venv` was built by the previous resolver.
- Version updates for `uv` stay out of `.github/dependabot.yml` on purpose:
  the scientific stack is bumped by hand with the full gate. Security updates
  must not be confused with that — they only reach for the first fixed
  release of a pin an advisory already covers.

Verify the path instead of assuming it. An update job is a run whose event is
`dynamic` and whose name starts with "Dependabot Updates"; read the failing
line rather than the run colour:

```bash
R=Freston1605/Poly-X-Microplastics
gh run list -R $R --event dynamic --limit 20 \
  --json databaseId,displayTitle,conclusion
# Success signal: a run whose log reaches `uv lock` and carries no
# `tool_version_not_supported`. Then:
gh run view -R $R <run-id> --log | grep -E 'tool_version_not_supported|uv lock' \
  || echo 'no version-gate error in this job'
```

An advisory must exist to schedule such a run, and it reads the configuration
on the default branch; a fix on a feature branch cannot be observed until it
lands on `main`. So record the run id you checked, not the intention to check
one. If the only available evidence is the queued check, say exactly that:
"queued after merge on the next advisory; success = the same job log without
`tool_version_not_supported` and a populated resolved set."

The dependency-review job is not a substitute for either: it audits the
resolved set a pull request **adds**, and it cannot see an advisory published
against a pin that is already on `main`. See #102.

## Six checks block a merge

A pull request merges only when six contexts are green **and** its branch is up
to date with `main`. The list sits on the `required_status_checks` rule of the
default-branch ruleset `Copilot review for default branch` (`25976878`). #105
filled that rule; it was already enabled with
`strict_required_status_checks_policy: true`, and nothing outside the array
below was changed.

| Context | Reported by | Why it is required |
| --- | --- | --- |
| `quality` | `ci.yml` → `quality` | The gate itself: the merge contract. |
| `test (3.12)` | `ci.yml` → `test`, matrix `3.12` | The gate runs the suite on 3.11 only. |
| `build` | `ci.yml` → `build` | The catalog inside the wheel. |
| `pre-commit` | `pre-commit.yml` → `pre-commit` | The hooks, without a local install. |
| `dependency-review` | `ci.yml` → `dependency-review` | A vulnerable addition is blocked. |
| `workflow-lint` | `ci.yml` → `workflow-lint` | actionlint and zizmor, before a run. |

Those one-line reasons, expanded: `quality` runs the catalog audit, the
palette rule, the full suite, ruff and ty; `build` loads the compiled catalog
from **inside** the wheel, which Hatchling can silently exclude because the
source tree cannot see it; `dependency-review` blocks a pull request that adds
a package with a fresh advisory, since an exact pin stops drift and not
disclosure.

The strict "branch must be up to date" policy stays on: without it, two
individually green pull requests can merge into a combination neither one
tested.

### The two traps that break this list

Both fail the same way: the merge button waits forever, and the run that
produced the green checks is not the reason.

- **A context is the job name as reported, not the workflow name.** Renaming a
  job drops its protection silently; a name no job produces leaves the check
  "Expected" on every pull request, including the one that would fix it.
- **`test (3.12)` embeds the matrix value.** Editing the Python matrix in
  `ci.yml` renames the check and leaves the old context required and absent,
  so the matrix edit and this list have to land together.

One more is worth knowing without being a trap today: a required check whose
job is skipped by an `if:` on a pull request never reports either. The only
conditional job here, `dependency-review`, is conditional on the event being
`pull_request` — the event that is gated — so a pull request always runs it.

### Reading it back

```bash
R=Freston1605/Poly-X-Microplastics
gh api repos/$R/rulesets/25976878 --jq '
  .rules[] | select(.type == "required_status_checks") | .parameters'
# {"strict_required_status_checks_policy":true,"do_not_enforce_on_create":false,
#  "required_status_checks":[{"context":"quality"},{"context":"test (3.12)"},
#  {"context":"build"},{"context":"pre-commit"},{"context":"dependency-review"},
#  {"context":"workflow-lint"}]}
```

### Re-applying it

`PUT` replaces the whole ruleset, so read it and send every other rule back
untouched — the Copilot review rule, `code_quality`, `deletion`,
`non_fast_forward` and the `pull_request` rule are siblings of this one, and a
hand-written body drops whichever ones it forgets. Rewriting only the one
array keeps them byte-identical:

```bash
R=Freston1605/Poly-X-Microplastics
gh api repos/$R/rulesets/25976878 \
  | jq '{name, target, enforcement, conditions, bypass_actors,
      rules: [.rules[]
        | if .type == "required_status_checks"
          then .parameters.required_status_checks = [
            {context: "quality"},
            {context: "test (3.12)"},
            {context: "build"},
            {context: "pre-commit"},
            {context: "dependency-review"},
            {context: "workflow-lint"}
          ]
          else . end]}' \
  | gh api --method PUT repos/$R/rulesets/25976878 --input -
```

Then read it back with the command above, and diff the rule array instead of
trusting the `200`: a context spelled `test (3.12)` with a different space, or
a job that was already renamed, is the failure this list cannot detect itself.
The contexts can be confirmed against a real run — `gh pr checks <n>` lists
exactly the names GitHub matches.

## Re-applying after a fork reset

In order:

1. `gh api repos/Freston1605/Poly-X-Microplastics --jq .permissions.admin` —
   confirm you are an admin. Everything below is a no-op without it.
2. Enable Actions on the fork (GitHub disables workflows on forks of public
   repositories; no setting below turns them on). Then set
   `sha_pinning_required`.
3. Create the `Protect release tags` ruleset, if the ruleset list does not
   already have it.
4. Turn on private vulnerability reporting, Dependabot alerts and security
   updates. Confirm each with its `GET`, and check that `required-version` in
   `pyproject.toml` is the uv that Dependabot bundles — otherwise the setting
   is on and every Python job fails (see "Security updates are only on when
   the uv pin is satisfiable").
5. Fill the six required checks, last — and only once the jobs they name run
   on `main` (see "Six checks block a merge").
6. Verify the whole set with the reads below — a `PUT` that returns `204` is
   not evidence that the value is what you sent.

```bash
R=Freston1605/Poly-X-Microplastics
gh api repos/$R/actions/permissions
gh api repos/$R/rulesets --jq '.[] | {id, name, target, enforcement}'
gh api repos/$R/rulesets/<tag-ruleset-id> --jq '.bypass_actors'
# The effective default-branch rules, as GitHub applies them — this is the
# read that shows the required-checks list actually reached `main`.
gh api repos/$R/rules/branches/main \
  --jq '.[] | select(.type == "required_status_checks") | .parameters'
gh api repos/$R/private-vulnerability-reporting
gh api repos/$R/automated-security-fixes
gh api repos/$R/vulnerability-alerts -i | head -1   # HTTP/2.0 204 = on
```

Expected: `sha_pinning_required: true`; two rulesets in this repository, one
`branch` and one `tag`; `bypass_actors` naming repository roles 2 and 5; the
six contexts with `strict_required_status_checks_policy: true`;
`{"enabled":true}`; `{"enabled":true,"paused":false}`; `204`.

## What this runbook cannot re-apply

- **PyPI Trusted Publishing.** The publisher is configured on pypi.org and is
  bound to the owner, repository and workflow filename. A fork under a new
  owner needs a new publisher there; no GitHub API call does it.
- **The repository's owner and visibility.** A private copy has no public
  advisory form and Dependabot alerts differ; the settings above assume a
  public repository.
- **The decision itself.** The reasons above are the record; if a switch is
  turned off, delete its row rather than leaving the table claiming a
  protection that is not there.

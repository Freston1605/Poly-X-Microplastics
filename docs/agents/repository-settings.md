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
| Ruleset `Protect release tags` | `creation` and `deletion` of `refs/tags/v*`, bypass for admin and maintain | Repository → Rules → Rulesets |
| Require actions to be pinned to a full-length commit SHA | on | Settings → Actions → General |
| Private vulnerability reporting | on | Settings → Code security |
| Dependabot alerts | on | Settings → Code security |
| Dependabot security updates | on | Settings → Code security |
| Required status checks | set by #105, not by this runbook | Default-branch ruleset |

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
| Rules | `creation`, `deletion` |
| Bypass | `RepositoryRole` 2 (maintain), 5 (admin), mode `always` |

GitHub's model is inverted with respect to the wording: a rule applies to
**everyone except** the bypass list. So "only maintainers can create a tag"
is the `creation` rule plus an admin/maintain bypass — not "allow maintainers"
plus an empty list, which would let anyone tag.

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
  "rules": [{"type": "creation"}, {"type": "deletion"}],
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

The dependency-review job is not a substitute for either: it audits the
resolved set a pull request **adds**, and it cannot see an advisory published
against a pin that is already on `main`. See #102.

## Required status checks belong to #105

This file deliberately does **not** carry the required-checks list, and this
change does not set it. The default-branch ruleset
`Copilot review for default branch` (`25976878`) already holds the rule with
`strict_required_status_checks_policy: true` and an **empty**
`required_status_checks` array; filling it is the merge contract, and it is
[#105](https://github.com/Freston1605/Poly-X-Microplastics/issues/105) —
apply that after the jobs it names exist.

When you re-apply the list from a fork reset, take the contexts from #105
rather than from memory. Two traps:

- A check is matched by the **job name as reported**. Renaming a job, or
  deleting one, silently drops its protection instead of failing loudly.
- The `test (3.12)` context embeds the matrix value. Editing the Python matrix
  renames the check and leaves the old context required and absent, so the
  matrix edit and the ruleset edit have to land together.

The strict "branch must be up to date" policy stays on: without it, two
individually green pull requests can merge into a combination neither one
tested.

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
   updates. Confirm each with its `GET`.
5. Land #105's required-checks list, last.
6. Verify the whole set with the reads below — a `PUT` that returns `204` is
   not evidence that the value is what you sent.

```bash
R=Freston1605/Poly-X-Microplastics
gh api repos/$R/actions/permissions
gh api repos/$R/rulesets --jq '.[] | {id, name, target, enforcement}'
gh api repos/$R/rulesets/<tag-ruleset-id> --jq '.bypass_actors'
gh api repos/$R/private-vulnerability-reporting
gh api repos/$R/automated-security-fixes
gh api repos/$R/vulnerability-alerts -i | head -1   # HTTP/2.0 204 = on
```

Expected: `sha_pinning_required: true`; two rulesets in this repository, one
`branch` and one `tag`; `bypass_actors` naming repository roles 2 and 5;
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

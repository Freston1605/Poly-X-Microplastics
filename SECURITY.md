# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 2.0.x | ✅ |
| < 2.0 | ❌ |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Email the maintainer at **cristofher.ferrada@pucv.cl** with:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept),
- the affected version(s).

You will receive an acknowledgement within 7 days and a fix or
mitigation plan within 30 days. Once a fix is released, the issue will
be disclosed publicly with credit to the reporter (unless you prefer
to remain anonymous).

## Scope

Poly-X is a desktop application that runs locally on the user's machine
and processes images the user provides. It does not expose a network
service, accept remote input, or store credentials. The main security
surface is the parsing of untrusted files (images, YOLO `.txt` labels,
dataset YAML, model `.pt` weights) and the HTML/PDF report generation.

Reports of denial-of-service via malformed inputs, path traversal in
report generation, or unsafe deserialization of model weights are in
scope. Vulnerabilities in third-party dependencies (PyTorch, Ultralytics,
PySide6, OpenCV) should be reported upstream; please mention the
affected Poly-X version so we can pin a fixed release.
# RUNE Release Process

This document describes the synchronized release process for the RUNE ecosystem.

## 1. Synchronized Versioning
RUNE follows [Semantic Versioning](https://semver.org/). While repositories are independent, we aim to keep major and minor versions synchronized across the core components:
- `rune` (Core)
- `rune-operator`
- `rune-ui`
- `rune-charts`
- `rune-docs`

## 2. Release Prerequisites
A release **MUST** only be initiated after:
1.  A Pull Request has been merged into the default branch (`main` or `master`).
2.  All **RuneGate** quality checks (97-100% coverage, SAST, CVE scans) are **GREEN**.
3.  The project metadata has been updated to the new version:
    - **Python:** `pyproject.toml` version field.
    - **Helm:** `Chart.yaml` (both `version` and `appVersion`).

## 3. The Tagging Workflow
Release automation is triggered by pushing a version tag.

```bash
# 1. Sync local environment
git checkout main
git pull origin main

# 2. Create the signed tag
git tag -a v0.x.y -m "Release v0.x.y"

# 3. Push the tag to trigger automation
git push origin v0.x.y
```

## 4. Automated Artifacts

| Component | Automated Actions |
| :--- | :--- |
| **`rune`** | Publishes to **PyPI**, creates GitHub Release, and auto-tags `rune-docs`. |
| **`rune-operator`** | Builds multi-arch Docker images to **GHCR**, creates GitHub Release. |
| **`rune-ui`** | Builds multi-arch Docker images to **GHCR**, creates GitHub Release. |
| **`rune-charts`** | Packages Helm charts and attaches them to a GitHub Release. |
| **`rune-docs`** | Builds static site Docker image to **GHCR**, creates GitHub Release. |

## 5. Security Gates (ML4)
Every release artifact is automatically:
- Scanned for CVEs using **Grype** and **Trivy**.
- Provenance-attested using **SLSA Level 3** (GitHub Attestations).
- Verified for license compliance.

Vulnerabilities above the threshold (CVSS 7.0+) will block the release automation until fixed or risk-accepted.

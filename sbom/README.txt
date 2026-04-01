SBOM artifacts for ai-benchmarks
================================

Standard location:
- This folder is the repository standard location for SBOM-related assets and notes.

What CI produces:
- CycloneDX SBOM (JSON): sbom/rune-image.cdx.json
- Grype SARIF scan report: sbom/grype-results.sarif

How it is generated:
- GitHub Actions workflow: .github/workflows/coverage.yml
- SBOM generator action: anchore/sbom-action
- SBOM vulnerability scanner action: anchore/scan-action (Grype)

Notes:
- Generated JSON/SARIF files are ignored by git and uploaded as workflow artifacts.
- SARIF is also uploaded to GitHub code scanning.

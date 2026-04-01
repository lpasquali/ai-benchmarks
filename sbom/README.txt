SBOM artifacts for ai-benchmarks
================================

Standard location:
- This folder is the repository standard location for SBOM-related assets and notes.

What CI produces:
- CycloneDX SBOM (JSON): sbom/rune-image.cdx.json
- Grype scan report (JSON): sbom/grype-results.json
- Trivy scan report (JSON): sbom/trivy-results.json
- Unified security summary: sbom/security-summary.txt
- Summary SARIF report: sbom/security-summary.sarif

How it is generated:
- GitHub Actions workflow: .github/workflows/coverage.yml
- SBOM generator tool: Syft container (`anchore/syft`)
- SBOM vulnerability scanner tools:
	- Grype container (`anchore/grype`)
	- Trivy container (`aquasec/trivy`)

Policy gate:
- Merge is blocked if any vulnerability with CVSS score > 8.8 is detected by either scanner.
- Results are normalized into one summary so scanner outputs are homogeneous in CI artifacts.

Notes:
- Generated JSON/SARIF files are ignored by git and uploaded as workflow artifacts.
- SARIF is also uploaded to GitHub code scanning.

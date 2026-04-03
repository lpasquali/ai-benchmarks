SBOM artifacts for rune
================================

Standard location:
- This folder is the repository standard location for SBOM-related assets and notes.

What CI produces:
- CycloneDX SBOM (JSON): sbom/rune-image.cdx.json
- Grype scan report (JSON): sbom/rune-grype.json
- Trivy scan report (JSON): sbom/rune-trivy.json

How it is generated:
- GitHub Actions workflow: .github/workflows/quality-gates.yml (job: RuneGate/Security/SBOM-and-CVE-Policy)
- SBOM generator tool: Syft container (`anchore/syft`)
- SBOM vulnerability scanner tools:
	- Grype container (`anchore/grype`)
	- Trivy container (`aquasec/trivy`)

Policy gate:
- Merge is blocked if any **fixable** vulnerability with CVSS score > 8.8 is detected.
- Unfixable vulnerabilities above threshold are logged as warnings and tracked via VEX register (security/vex-register.md).
- Results from both scanners are stored as workflow artifacts (sbom-security-outputs, 14-day retention).

Notes:
- Generated JSON files are ignored by git and uploaded as workflow artifacts.
- SARIF is uploaded to GitHub Security tab via the CodeQL workflow.

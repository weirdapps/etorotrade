# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities to the repository maintainers through GitHub's private vulnerability reporting feature or by creating a private security advisory.

## Known Security Considerations

### GitHub Actions

#### Claude Code Action (Beta) - CURRENTLY DISABLED
- **Issue**: The `anthropics/claude-code-action@beta` cannot be pinned to a specific commit SHA as it's in beta
- **Risk Level**: Medium (Unacceptable without SHA pinning)
- **Status**: **DISABLED** - Workflow has been disabled to comply with security policy
- **Resolution**: 
  - The workflow file has been disabled (claude.yml shows security notice only)
  - Original workflow preserved in `.github/workflows/claude.yml.disabled`
  - Will be re-enabled once a stable version with SHA pinning is available
- **Temporary Enable Process** (if risk is accepted):
  1. Rename `.github/workflows/claude.yml.disabled` to `claude.yml`
  2. Document security exception with business justification
  3. Implement all mitigations (environment approval, trigger validation, timeout)
- **Tracking**: Monitor for stable releases at https://github.com/anthropics/claude-code-action/releases

#### All Other Actions
All other GitHub Actions are pinned to specific commit SHAs for security:
- `actions/checkout`: SHA `11bd71901bbe5b1630ceea73d27597364c9af683` (v4.2.2)
- `actions/setup-python`: SHA `f677139bbe7f9c59b41e40162b753c062f5d49a3` (v5.2.0)
- `actions/cache`: SHA `6849a6489940f00c2f30c0fb92c6274307ccb58a` (v4.1.2)
- `codecov/codecov-action`: SHA `b9fd7d16f6d7d1b5d2bec1a2887e65ceed900238` (v4.6.0)
- `actions/upload-artifact`: SHA `b4b15b8c7c6ac21ea08fcf65892d2ee8f75cf882` (v4.4.3)

### Docker Security

The Docker container implements multiple security layers:
1. **Non-root execution**: Runs as `appuser`
2. **Read-only filesystem**: Application code is immutable (chmod 444/555)
3. **No embedded secrets**: Configuration must be mounted externally
4. **Minimal base image**: Uses `python:3.11-slim`
5. **Health checks**: Container health monitoring enabled

### Dependencies

- All Python dependencies are specified with exact versions in `requirements.txt`
- Regular dependency updates are performed
- Security scanning via Dependabot is enabled

## Security Best Practices

1. **Never commit secrets**: Use GitHub Secrets for sensitive data
2. **Review PRs carefully**: Especially those modifying workflows or Docker configuration
3. **Keep dependencies updated**: Regular updates for security patches
4. **Use environment protection**: Required approvals for production deployments
5. **Monitor SonarCloud**: Regular review of security hotspots and vulnerabilities

## Compliance

This project uses SonarCloud for continuous security analysis. Current security posture:
- Security hotspots are reviewed and addressed regularly
- Known issues are documented with risk assessments
- Mitigations are implemented where direct fixes aren't possible
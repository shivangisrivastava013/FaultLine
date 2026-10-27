# Security Policy & Responsible Disclosure

## 1. Supported Versions

Security updates and vulnerability remediation are actively applied to the default branch (`main`):

| Version | Supported |
| :--- | :--- |
| `main` (Latest) | :white_check_mark: |
| Development branches | :x: |

---

## 2. Responsible Disclosure

If you discover a potential security vulnerability within FaultLine, please report it responsibly rather than opening a public issue on GitHub.

### Reporting Process:
1. **Direct Communication**: Send details of the vulnerability to [shivangisrivastava013@gmail.com](mailto:shivangisrivastava013@gmail.com) with the subject line `[SECURITY] FaultLine Vulnerability Report`.
2. **Details to Include**:
   - Description of the vulnerability and attack vector.
   - Minimal reproducible proof of concept (PoC) or script.
   - Potential impact on model serving, database integrity, or administrative endpoints.
3. **Response Timeline**:
   - Initial acknowledgement within 48 hours.
   - Remediation patch and verification within 7 business days.
   - Public release notes credited to the reporter once mitigated.

---

## 3. Local Development Credentials Notice

> [!WARNING]
> **Default Credentials Isolation:**
> The default admin API key (`ADMIN_API_KEY=faultline_dev_secret_key`), PostgreSQL database credentials (`POSTGRES_USER=faultline`, `POSTGRES_PASSWORD=faultline_password`), and Grafana administrative login (`admin` / `admin`) provided in `docker-compose.yml` and `.env.example` are strictly designed for local developer sandbox execution and reproducible simulation benchmarks.

### Staging & Production Security Requirements:
1. **Administrative Authentication**:
   - Administrative endpoints (`/rollback`, `/admin/activate-model`, `/admin/inject-fault`) require constant-time `X-API-Key` verification (`secrets.compare_digest`).
   - For non-local environments, generate a high-entropy secret (e.g. `openssl rand -hex 32`) and inject it via environment variables or a cloud secrets manager (e.g. AWS Secrets Manager, HashiCorp Vault).
2. **Network Isolation**:
   - Database and internal Prometheus endpoints should reside within private VPC subnets with no public ingress.
   - External traffic to the prediction API must be terminated behind an API Gateway or Application Load Balancer with TLS/HTTPS.
3. **AI Provider Credentials**:
   - Optional LLM API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`) must never be hardcoded or committed to version control. FaultLine strictly ignores `.env` files via `.gitignore`.

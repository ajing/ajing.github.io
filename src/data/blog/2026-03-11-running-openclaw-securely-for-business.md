---
author: Jing Lu
pubDatetime: 2026-03-11T06:00:00-07:00
title: "Running OpenClaw Securely for Your Small Business: A Practical Guide"
featured: true
draft: true
tags:
  - AI
  - Security
  - OpenClaw
  - LLM
  - Agents
description: "OpenClaw is the fastest-growing AI agent, but can you use it to run a business? This post investigates the security landscape — from CVE-2026-28472 to prompt injection — and presents a practical strategy for solo operators who want maximum automation without losing sleep."
---

> OpenClaw has 250K+ GitHub stars and can automate nearly anything on your computer. But if you want to build a business around it, you need to understand what can go wrong — and how to make it survivable.

---

## What Is OpenClaw?

OpenClaw is an open-source autonomous AI agent that runs locally on your machine. It connects LLMs (Claude, GPT, DeepSeek) to your files, messaging apps, APIs, and shell — enabling it to actually *do things*, not just answer questions.

| Attribute | Detail |
|---|---|
| **Architecture** | Persistent Node.js service, local gateway |
| **Capabilities** | Shell commands, file I/O, web automation, email, calendar, messaging |
| **Extensibility** | "Skills" from ClawHub marketplace |
| **Model Support** | Model-agnostic (Claude, GPT, DeepSeek, local models) |
| **Origin** | Created by Peter Steinberger (now at OpenAI); managed by independent foundation |

The promise is compelling: a personal AI employee that handles your inbox, writes code, manages your calendar, and even makes purchases. But with great power comes great attack surface.

---

## The Security Landscape (It's Not Pretty)

### Critical CVEs

**CVE-2026-28472 — "ClawJacked" (CVSS 9.2)**

The most severe vulnerability to date. Malicious websites could silently hijack your local OpenClaw agent via WebSocket. Browsers don't block WebSocket connections to `localhost`, and OpenClaw exempted localhost from rate limiting — so an attacker could brute-force the password at hundreds of attempts per second, register as a trusted device, and gain full control.

**Impact:** Full workstation compromise, data exfiltration, access to private repos and cloud APIs. **Status:** Patched in v2026.2.2+.

**CVE-2026-25253 — Control UI WebSocket RCE**

Remote code execution via the control UI. **Patched in v2026.1.29+.**

### The "ClawHavoc" Supply Chain Campaign

In early 2026, security researchers discovered **hundreds of malicious skills on ClawHub** (OpenClaw's public skill registry). These skills established reverse shells, exfiltrated SSH keys and API tokens, and stole browser session cookies. Microsoft and Cisco both published warnings.

### The Fundamental Problem: Prompt Injection

Every other vulnerability can be patched. Prompt injection **cannot** — it's architectural.

LLMs process instructions and data in the same channel. There's no hardware-level separation between "this is a trusted system prompt" and "this is text from a webpage the agent just read." A malicious instruction embedded in a document, email, or webpage can hijack the agent's actions.

Even OpenAI's CISO calls it "an unsolved security problem." Anthropic reports Claude Sonnet 4.5 blocks 94% of injection attacks in agentic settings — impressive, but that remaining 6% is nonzero.

**The practical implication:** You can't prevent every attack. You design your system assuming attacks WILL succeed and limit the damage.

---

## The Strategy: Secure Enough for a Solo Business

The key insight: **not all automation is equally dangerous.** Classify your agent's actions:

| Safe (fully automated) | Risky (auto + logged) | Dangerous (needs your OK) |
|---|---|---|
| Read files & web pages | Send emails/messages | Access passwords/API keys |
| Summarize & draft text | Create/edit files | Big purchases (>$20) |
| Search & answer questions | Schedule events | Delete important data |
| Write & run code (sandboxed) | Research & add to cart | |
| Compare prices, find deals | Small auto-purchases (≤$20) | |

**~90% of the useful automation is in the "Safe" column.** You keep all of it.

---

## The 6-Layer Defense Stack

### Layer 1: Container Isolation

Run OpenClaw in Docker, not directly on your machine. If the agent gets compromised, it's trapped in a container.

```bash
docker run -d \
  --name openclaw \
  --restart unless-stopped \
  -e OPENCLAW_AUTH_TOKEN="$(openssl rand -hex 32)" \
  -v ~/openclaw-workspace:/workspace \
  -p 127.0.0.1:18789:18789 \
  --read-only \
  --tmpfs /tmp \
  --cap-drop ALL \
  openclaw/openclaw:latest
```

Key flags: `127.0.0.1` binds to localhost only; `--read-only` prevents container self-modification; `--cap-drop ALL` removes Linux privileges.

### Layer 2: Sandbox Mode

Enable OpenClaw's built-in sandbox so code runs freely but can't escape:

```bash
openclaw config set agents.defaults.sandbox.mode all
```

The sandbox lets the agent write and run Python scripts, Node servers, build tools — whatever — but it can't reach your host system. If injected code tries to `curl` your secrets or access files outside the workspace, the sandbox blocks it.

### Layer 3: Agent Integrity Monitoring

**Install [ClawSec](https://github.com/prompt-security/clawsec)** (by Prompt Security / SentinelOne):

```bash
openclaw install prompt-security/clawsec
```

ClawSec monitors for tampering with your agent's `SOUL.md` and `IDENTITY.md` files (drift detection + auto-restore), polls NIST NVD for relevant CVEs, and verifies skill integrity via SHA256 checksums. It's free and made by a credible team (Prompt Security was acquired by SentinelOne for ~$250M).

**But ClawSec alone is not enough.** It doesn't detect prompt injection, doesn't audit skills before installation, and doesn't monitor for active threats. You also want:

| Tool | What It Covers |
|---|---|
| **[SecureClaw](https://github.com)** | 55 automated audit + hardening checks, OWASP/MITRE alignment |
| **[openclaw-security-monitor](https://github.com)** | Real-time threat detection (ClawHavoc, AMOS stealer, memory poisoning) |
| **[LLM Guard](https://github.com/protectai/llm-guard)** | Input/output scanning for prompt injection (15 input + 20 output scanners) |

### Layer 4: Strong Model Selection

Use the strongest reasoning model you can afford. Injection resistance scales with model capability:

| Model | Agentic Injection Resistance |
|---|---|
| Claude Sonnet 4.5 | 94% (with detection enabled) |
| GPT-5.3 | Strong (trained to refuse malicious requests) |
| DeepSeek V3 | Weaker injection resistance (cheaper, trade-off) |

### Layer 5: Financial Guardrails

For purchasing automation, **the real guardrail is the card limit, not the SOUL.md.** Card limits are enforced by the bank — the LLM can't bypass them.

1. Get a **virtual/prepaid card** (Privacy.com, Revolut, or your bank's virtual card)
2. Set a **spending limit** ($200/month)
3. Lock to **specific merchants** if your provider supports it
4. Give **only this card** to the agent — never your real card

Even in the worst case (injection + all software defenses bypassed), your maximum loss = the card's monthly limit.

### Layer 6: Red Teaming

Test your defenses monthly with:

- **[Garak](https://github.com/NVIDIA/garak)** (NVIDIA) — LLM vulnerability scanner
- **[PyRIT](https://github.com/Azure/PyRIT)** (Microsoft) — Automated AI red teaming

---

## OpenAI Codex vs OpenClaw: Which Is More Secure?

If you're considering the OpenAI Pro tier ($200/mo) with Codex, here's how the security models compare:

| Dimension | OpenAI Codex | OpenClaw |
|---|---|---|
| **Sandbox** | Cloud containers, no internet access during tasks, sealed environment | Docker + built-in sandbox (you set it up yourself) |
| **Who manages security** | OpenAI | You |
| **Prompt injection defense** | Model-level (trained to refuse), sealed sandbox | Manual: LLM Guard + ClawSec + model choice |
| **Skill/plugin risk** | OpenAI-controlled ecosystem | ClawHub (public, malicious skills found) |
| **Scope** | Software engineering + coding tasks | Everything (email, calendar, files, messaging, purchases, shell) |
| **Data privacy** | Data goes through OpenAI's cloud | Fully local (except LLM API calls) |
| **Cost** | $200/mo flat | Free + LLM API costs ($3-200/mo depending on model) |

**The honest comparison:** Codex is *more secure by default* for coding tasks — OpenAI handles the sandboxing, and the containers have no internet access. But Codex is mainly a coding agent. OpenClaw does *everything else* (messaging, scheduling, purchases, file management).

### The Hybrid Approach (Best of Both Worlds)

Use **Codex for coding tasks** (secure cloud sandbox, OpenAI manages infrastructure) and **OpenClaw for everything else** (messaging, calendar, purchases, file management) running locally in Docker.

### Lowering the Monthly Cost

The $200/mo Pro tier may be more than you need. Alternatives:

| Approach | Monthly Cost | Trade-off |
|---|---|---|
| **OpenAI Pro ($200)** | $200 | Full Codex + unlimited ChatGPT Pro |
| **OpenAI Plus ($20) + API** | $20 + usage | ChatGPT Plus for chat; API for Codex-mini at $0.25/M input tokens |
| **OpenClaw + DeepSeek API** | ~$3-10/mo | 10-50x cheaper than GPT; weaker injection resistance |
| **OpenClaw + local model (Ollama)** | $0 (electricity only) | Free but needs 16GB+ RAM; weakest injection resistance |
| **OpenClaw + Claude API** | ~$50-200/mo | Best injection resistance (94%); moderate cost |

**Recommended for a budget-conscious solo operator:**

- **ChatGPT Plus ($20/mo)** for occasional deep research and Codex tasks
- **OpenClaw + Claude API via Anthropic** for daily automation (~$50-100/mo depending on usage)
- **Total: ~$70-120/mo** — much less than $200, with better automation breadth

Or if you want to go even cheaper:

- **OpenClaw + DeepSeek V3** for most tasks (~$3-10/mo)
- **Claude API only for sensitive/risky tasks** (purchasing, messaging) where injection resistance matters (~$20-30/mo)
- **Total: ~$25-40/mo** — with a conscious trade-off on injection resistance for routine tasks

---

## The SOUL.md Security Rules

Configure your agent's behavior policy:

```markdown
## Security Rules (NEVER override these)

1. You may freely write, edit, and run code inside the sandbox
2. NEVER access, display, or transmit API keys, passwords, or tokens
3. NEVER delete important data without asking me first
4. NEVER send messages to people outside my approved contact list
5. For purchases:
   - You may research, compare prices, and add items to cart freely
   - For items ≤$20 from approved merchants: you may auto-purchase
   - For items >$20: send me a summary (item, cost, merchant) and wait for "approved"
6. For everything in my workspace — read, write, and execute freely
7. If a task requires access outside the sandbox, explain what you need and why
```

Remember: SOUL.md rules are speed bumps, not walls. Prompt injection can bypass them. That's why you layer Docker + sandbox + card limits + monitoring on top.

---

## Weekend Setup Checklist

### Day 1: Foundation
- [ ] Install Docker on your old computer
- [ ] Run OpenClaw in Docker with security flags (see above)
- [ ] Enable sandbox mode (`agents.defaults.sandbox.mode all`)
- [ ] Set a strong auth token
- [ ] Configure your SOUL.md security rules

### Day 2: Defense Layers
- [ ] Install ClawSec (`openclaw install prompt-security/clawsec`)
- [ ] Set up a virtual/prepaid card for agent purchases
- [ ] Store API keys in `.env` with `chmod 600`
- [ ] Disable ClawHub, don't install random skills
- [ ] Run `openclaw security audit --deep --fix`

### Ongoing (5 min/week)
- [ ] Glance at logs daily (2 minutes)
- [ ] Update weekly: `docker pull openclaw/openclaw:latest`
- [ ] Rotate auth tokens monthly
- [ ] Run Garak vulnerability scan monthly

---

## Conclusion

OpenClaw is powerful enough to be genuinely useful for a small business — automating email, scheduling, code, research, and even purchasing. It's also powerful enough to be genuinely dangerous if deployed carelessly.

The security posture for a solo operator is not enterprise-grade hardening. It's **practical risk management:**

1. **Contain the blast radius** (Docker + sandbox)
2. **Monitor integrity** (ClawSec + security-monitor)
3. **Use bank-enforced financial limits** (virtual cards)
4. **Use the strongest model you can afford** (injection resistance scales with capability)
5. **Accept the residual risk** — prompt injection can't be fully solved, but for a small business that isn't a high-value target, the practical risk is manageable

You keep ~95% of the automation. The 5% you gate is the stuff that would ruin your week if it went wrong.

---

## References

1. **OWASP Top 10 for LLM Applications** (2025) — Prompt injection as #1 vulnerability
2. **OWASP Top 10 for Agentic Applications** (2026) — Agent Goal Hijack (ASI01)
3. **CVE-2026-28472** — "ClawJacked" authentication bypass, CVSS 9.2
4. **CVE-2026-25253** — Control UI WebSocket RCE
5. **ClawHavoc Campaign** — Malicious skills on ClawHub (Microsoft, Cisco reports)
6. **ClawSec** — prompt-security/clawsec, security skill suite by Prompt Security / SentinelOne
7. **SecureClaw** — 55 automated audit checks aligned with OWASP/MITRE
8. **LLM Guard** — Protect AI, open-source input/output scanning
9. **Garak** — NVIDIA, LLM vulnerability scanner
10. **PyRIT** — Microsoft AI Red Team, automated adversarial testing
11. **Design Patterns for Securing LLM Agents** (June 2025) — Dual LLM, Action-Selector, Plan-Then-Execute patterns
12. **Google DeepMind CaMeL** (April 2025) — Refined Dual LLM pattern for prompt injection defense
13. **Anthropic Transparency Hub** — Claude Sonnet 4.5: 94% injection prevention in agentic MCP scenarios
14. **Simon Willison** — Dual LLM pattern, "lethal trifecta" of AI agent risks
15. **Slowmist** — openclaw-security-practice-guide, Agentic Zero-Trust Architecture

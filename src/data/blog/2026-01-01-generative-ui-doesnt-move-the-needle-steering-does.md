---
author: Jing Lu
pubDatetime: 2026-01-01T00:00:00Z
title: "Generative UI Doesn't Move the Needle—Steering Does"
featured: true
draft: false
tags:
  - AI
  - UX
  - LLM
  - Product
description: "After shipping multiple generative UI features, I've concluded that the sophistication of AI-generated interfaces often doesn't translate to user benefit—but steering does."
---

After working on multiple generative UI projects over the past year, I've reached a somewhat counterintuitive conclusion: **the sophistication of AI-generated interfaces often doesn't translate to meaningful user benefit**.

When OpenAI released Canvas in late 2024 and Google demoed Gemini generating Flutter UIs on the fly, many of us believed we had reached the inflection point for dynamic, AI-generated interfaces. But after shipping several generative UI features to production and observing real user behavior, I've become skeptical of this vision—at least in its current form.

---

## Three Approaches to Generative UI

### 1. Full HTML/CSS/JS Generation

The LLM generates complete web pages or interactive widgets from scratch.

**Pros:** Maximum flexibility, rich visualizations  
**Cons:** High latency, inconsistent quality, security concerns

### 2. Constrained Widget Composition

The LLM selects and composes from a predefined catalog of trusted UI components. This is the approach used by Vercel's AI SDK and Google's [A2UI (Agent-to-UI)](https://a2ui.org) project.

Key A2UI design principles:
- **Declarative, not executable** — JSON description, not code
- **Flat list with ID references** — Easy incremental updates
- **Two-layer data model** — Separate UI structure from data
- **Client-controlled rendering** — Agent requests, client owns styling

**Pros:** Faster, consistent visual language, security by design  
**Cons:** Limited to predefined components

### 3. Offline Template Generation + Runtime Selection

Use LLMs during development to generate templates, then at runtime simply select the appropriate template.

**Pros:** Lowest latency, highest quality, predictable UX  
**Cons:** No true "generation" at runtime

---

## What I Learned After Deploying These Systems

### For Information-Seeking Tasks

An interactive visualization *can* help users understand algorithms better than text. But in practice, users often find that a simple code snippet—which they can copy, modify, and run—is more valuable. The "wow factor" wears off quickly.

### For Widget-Based Interaction

We expected users would prefer tapping buttons over typing. What we found: **the generated widgets were rarely more helpful than well-formatted markdown text**.

Why? Users can always refine their request through natural language. The flexibility of saying "actually, make it for next Tuesday instead" beats clicking through a date picker.

### For Template-Based UIs

By front-loading generation to development time—where you can iterate with LLMs, run aesthetic agents, add human review—you get production-quality templates without runtime latency.

---

## What's Actually Valuable: Steering

If we step back and ask *why* we need UI beyond text, the answer isn't "to make things prettier." It's **to enable better steering of the AI**.

Consider: An LLM processes your request through 15 reasoning steps. At step 7, it makes a subtle mistake. What you need is a way to:

1. **Inspect intermediate steps** easily
2. **Point to a specific step** and say "this is where you went wrong"
3. **Redirect the model** from that point

This is *steering*—and it's where UI can genuinely add value.

---

## The Steering Test for Generative UI

> **Does this UI help the user steer the model more effectively?**

### UI That Passes the Test

- **Inline editing with selection** — Cursor's `Cmd+K`, Canvas's text selection
- **Plan review with annotations** — Antigravity IDE's "Planning Mode"
- **Quick-action shortcuts** — Canvas's one-click writing/code buttons
- **Artifacts as manipulable objects** — Claude's side panel artifacts
- **Agent supervision with checkpoints** — Monitoring progress, intervening early
- **Context reference UI** — Cursor's `@` mentions and codebase indexing

### UI That Fails the Test

- Fancy animated visualizations that don't accept input
- Widget carousels that constrain rather than expand options
- Beautiful reports that are essentially read-only

---

## Conclusion

Generative UI is technically fascinating, but technical sophistication isn't the same as user value. The right question isn't "can we generate this UI?" but "does this UI help users communicate with and steer the AI more effectively?"

The most impactful AI interfaces won't be the most visually impressive ones—they'll be the ones that make human-AI collaboration more effective.

---

## References

- Mayer, R. E. (2009). *Multimedia Learning* (2nd ed.). Cambridge University Press.
- Nielsen Norman Group. (2023). "Chatbots: How to Design Conversational UI"
- OpenAI. (2024). "Introducing Canvas"
- Google. (2025). "[Introducing A2UI](https://developers.googleblog.com/introducing-a2ui-an-open-project-for-agent-driven-interfaces/)"


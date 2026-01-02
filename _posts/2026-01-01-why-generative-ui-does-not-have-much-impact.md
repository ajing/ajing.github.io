---
layout: post
title: "Generative UI Doesn't Move the Needle—Steering Does"
date: 2026-01-01
categories: [AI, UX, LLM]
---

# Generative UI Doesn't Move the Needle—Steering Does

After working on multiple generative UI projects over the past year, I've reached a somewhat counterintuitive conclusion: **the sophistication of AI-generated interfaces often doesn't translate to meaningful user benefit**.

When OpenAI released Canvas in late 2024 and Google demoed Gemini generating Flutter UIs on the fly, many of us (myself included) believed we had reached the inflection point for dynamic, AI-generated interfaces. The vision was compelling: interfaces that adapt in real-time to user needs, powered by LLMs that understand context and intent. But after shipping several generative UI features to production and observing real user behavior, I've become skeptical of this vision—at least in its current form.

---

## Three Approaches to Generative UI

From my experience and observing the industry, generative UI implementations fall into three main categories:

### 1. Full HTML/CSS/JS Generation

The most flexible approach: the LLM generates complete web pages or interactive widgets from scratch. Think of the rich, interactive reports you see in products like Perplexity's research reports or ChatGPT's data analysis outputs.

**Pros:**
- Maximum flexibility and expressiveness
- Can create highly engaging, data-rich visualizations
- Supports integration with maps, charts, audio, and interactive elements

**Cons:**
- High latency (generating thousands of tokens)
- Inconsistent quality and styling
- Security considerations with arbitrary code execution

### 2. Constrained Widget Composition

Instead of generating raw code, the LLM selects and composes from a predefined library of widgets. This is the approach used by Vercel's AI SDK for React and similar frameworks. The system maintains two data structures: a user data model (current context and data) and a widget state (tracking user selections across turns).

**Pros:**
- Faster than full generation
- Consistent visual language
- Safer—no arbitrary code execution

**Cons:**
- Limited to predefined components
- Can feel rigid compared to free-form generation

### 3. Offline Template Generation + Runtime Selection

The most latency-optimized approach: use LLMs during development to generate and refine templates offline (with as many agent passes, human reviews, and iterations as needed), then at runtime simply select the appropriate template based on user intent.

**Pros:**
- Lowest latency at serving time
- Highest quality (templates are pre-validated)
- Predictable user experience

**Cons:**
- No true "generation" at runtime
- Limited to pre-anticipated use cases

---

## The Disappointing Reality

Here's what I've observed after deploying these systems: **users don't benefit as much as we expected**.

### For Information-Seeking Tasks

Yes, an interactive visualization of bubble sort *can* help users understand the algorithm better than a text explanation. But in practice, users often find that a simple Python code snippet in a code block—which they can copy, modify, and run themselves—is more valuable. The "wow factor" of a generated animation wears off quickly, and users gravitate toward the most *useful* format, not the most *impressive* one.

Research supports this observation. Studies on multimedia learning (Mayer, 2009) show that additional visual elements only help when they reduce cognitive load—extraneous visuals can actually *increase* cognitive load and hurt comprehension.

### For Widget-Based Interaction

When we deployed constrained widget UIs, we expected users would prefer tapping buttons and interacting with cards over typing. What we found: **the generated widgets were rarely more helpful than well-formatted markdown text**.

Why? Users can always refine their request through natural language. The flexibility of saying "actually, make it for next Tuesday instead" beats clicking through a date picker that the AI happened to generate. The widget becomes a *constraint* on the user journey ("you should pick from these three options") rather than an *enhancement*. This aligns with findings from Nielsen Norman Group's research on chatbot UX: users often prefer open-ended text input over constrained choices because it gives them more control over the conversation.

To be fair, constraints aren't inherently bad. If you're building a gardening app, you probably *don't* want users asking about house renovation—constrained widgets can helpfully guide users toward supported workflows. But this is a product decision about scope, not a user experience win from generative UI. You could achieve the same guardrails with a well-designed traditional interface or prompt engineering on the backend.

### For Template-Based UIs

The template approach essentially optimizes the delivery of predetermined interfaces. It's useful engineering, but it's not really "generative" in a meaningful sense. Users experience it as the same predictable UI they always had—just served faster.

---

## What's Actually Valuable: Steering

If we step back and ask *why* we even need UI beyond text, voice, and video, the answer isn't "to make things prettier." It's **to enable better steering of the AI**.

Consider this scenario: An LLM processes your request through 15 reasoning steps. At step 7, it makes a subtle mistake—perhaps misinterpreting an ambiguity in your request. The final output is wrong, but you can't tell *where* it went wrong by reading the final answer.

What you need is a way to:
1. **Inspect intermediate steps** easily
2. **Point to a specific step** and say "this is where you went wrong"
3. **Redirect the model** from that point

This is the kind of interaction that text alone struggles with. You want to click on step 7, highlight the problematic assumption, and tell the model to reconsider. This is *steering*—and it's where UI can genuinely add value.

OpenAI's Canvas is actually a good example of steering-focused UI: you can select text, request specific changes, and iterate on portions of the output. Anthropic's Claude artifacts serve a similar purpose—letting you work with code or documents as objects you can manipulate, not just linear chat responses.

---

## The Steering Test for Generative UI

Based on this experience, I now apply a simple heuristic when evaluating generative UI features:

> **Does this UI help the user steer the model more effectively?**

If the answer is no—if the UI is purely cosmetic or just a fancier way to display information that text would convey equally well—then it's unlikely to deliver lasting user value, no matter how impressive the technology behind it.

Examples of UI that *passes* the steering test:

- **Inline editing with selection** — Cursor's `Cmd+K` lets you select code and request targeted edits with natural language, keeping you in control of exactly what changes. OpenAI's Canvas does the same for writing: select text, request specific changes, and iterate on portions without regenerating everything.

- **Plan review with annotations** — Google's Antigravity IDE has a "Planning Mode" where the AI generates an implementation plan before executing. Users can highlight specific sections and add inline comments like "Don't use raw SQL; use the existing ORM wrapper" or "Make sure the profile image is circular." The AI refines the plan based on these annotations before proceeding. This is steering in its purest form: reviewing intermediate reasoning and redirecting before costly execution.

- **Quick-action shortcuts for common intents** — Canvas provides one-click buttons for writing ("adjust reading level", "add polish", "make shorter") and code ("add comments", "fix bugs", "port to language"). These capture the most frequent steering actions and reduce friction.

- **Artifacts as manipulable objects** — Claude's artifacts render code and documents as interactive objects in a side panel, not just text in a chat. You can iterate on them, copy portions, and request targeted modifications—treating the output as a working document rather than a one-shot response.

- **Agent supervision with checkpoints** — Antigravity's Agent Manager lets you run multiple AI agents in parallel while monitoring their progress. Agents produce verifiable artifacts (screenshots, browser recordings, implementation plans) at each step, so you can intervene before they go too far down a wrong path.

- **Context reference UI** — Being able to quickly point the model to relevant files, documentation, or previous conversation turns. Cursor's `@` mentions and codebase indexing let you steer by saying "look at this file" rather than copy-pasting context manually.

Examples of UI that *fails* the steering test:
- Fancy animated visualizations that look impressive but don't accept input
- Widget carousels that constrain rather than expand user options
- Beautiful reports that are essentially read-only

---

## Conclusion

Generative UI is a technically fascinating capability, but technical sophistication isn't the same as user value. After working in this space, I've concluded that the right question isn't "can we generate this UI?" but "does this UI help users communicate with and steer the AI more effectively?"

The most impactful AI interfaces won't be the most visually impressive ones—they'll be the ones that make the collaboration between human and AI more effective. Sometimes that means a simple text box with good affordances for editing and refinement. Sometimes it means structured views that make intermediate reasoning inspectable.

In the end, this isn't a technology question. It's a question about communication: **what interface makes it easiest for users to express intent and for the AI to respond appropriately?** When we keep that question at the center, we build better products—even if they're less flashy than what pure technological capability would allow.

---

## References

- Mayer, R. E. (2009). *Multimedia Learning* (2nd ed.). Cambridge University Press.
- Nielsen Norman Group. (2023). "Chatbots: How to Design Conversational UI"
- OpenAI. (2024). "Introducing Canvas"
- Vercel. (2024). "AI SDK: Generative UI"
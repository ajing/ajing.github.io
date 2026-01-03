---
author: Jing Lu
pubDatetime: 2025-05-03T00:00:00Z
title: "UI Representation and Action Execution for Generative UI"
featured: false
draft: false
tags:
  - AI
  - UI
  - LLM
  - Generative
description: "Exploring structured UI representation using JSON Schema, and how to implement action handlers for generative UI systems."
---

## 1. Structured UI Representation

Large models excel at generating structured UI representations. Instead of verbose HTML/JSX, we use a compact JSON component tree that's easy to parse and validate. This approach offers several advantages:

- **Type Safety**: Using JSON Schema ensures valid component structures
- **Framework Agnostic**: The same representation can be compiled to different frameworks
- **Efficient Parsing**: Compact structure reduces token usage and processing overhead

OpenAI's Structured Outputs and function-calling mode guarantee that the model's responses conform to our schema, making the UI generation process reliable and predictable.

---

## 2. Component Schema and UI Structure

```json
{
  "$id": "https://example.com/schemas/component.json",
  "type": "object",
  "required": ["type"],
  "properties": {
    "id":        { "type": "string" },
    "type":      { "enum": ["Container","Text","Button","Image"] },
    "props":     { "type": "object" },
    "children":  { "type": "array", "items": { "$ref": "#" } }
  }
}
```

This schema defines the core structure of our UI components. Each component has:
- A unique identifier
- A specific type from a controlled set
- Custom properties
- Optional child components

---

## 3. UI Action and Interaction

### 3.1 Inspecting UI State

```python
genui.describe()
```

```text
// Instance (truncated)
{
  "id": "root",
  "type": "Container",
  "props": { "layout": "vertical", "padding": "spacingM" },
  "children": [
    {
      "id": "event_title",
      "type": "Text",
      "props": { "text": "Welcome!", "variant": "heading" }
    },
    {
      "id": "get_started_btn",
      "type": "Button",
      "props": { "label": "Get started", "action": "onboard" }
    }
  ]
}
```

The `describe()` method provides a complete snapshot of the current UI state, enabling precise control over the interface.

### 3.2 Modifying UI Elements

```python
# Update component properties
genui.update(element_id="event_title", style={"variant": "heading2"})
```

```text
✓ Style modification succeeded
```

Updates are applied directly to the component tree, ensuring type safety and predictable rendering.

### 3.3 Handling User Actions

```python
genui.action(click_button="get_started_btn")
```

```text
✓ Button click succeeded
```

Action handlers map user interactions to specific behaviors, creating a clear connection between UI elements and their functionality.

---

## 4. Implementation and Best Practices

To implement a robust Generative UI system:

1. **Define Clear Schemas**: Create comprehensive JSON schemas for all component types
2. **Implement Action Handlers**: Map UI events to specific behaviors
3. **Maintain State**: Keep track of UI changes and user interactions
4. **Validate Inputs**: Ensure all modifications conform to the schema

For production use, consider:
- Using uiSchema for layout-specific properties
- Implementing framework-agnostic compilation with tools like Mitosis
- Setting up analytics to track user interactions
- Creating a feedback loop to improve the model's UI generation

---

## References

- [Generative UI: Building Dynamic Interfaces with LLMs and AI](https://medium.com/@mehdi-zare/generative-ui-building-dynamic-interfaces-with-llms-and-ai-b515d943b9aa)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [React JSON Schema Form - uiSchema](https://rjsf-team.github.io/react-jsonschema-form/docs/api-reference/uiSchema/)
- [Mitosis - Build framework-agnostic components](https://mitosis.builder.io/)


# UI representation and action execution for Generative UI

## 1 Why Generative UI needs structure

Large models excel at emitting *code‑like* text, but HTML/JSX can be verbose and error‑prone; a compact JSON component tree keeps token counts low and is trivial to parse ([Medium][1]).
OpenAI’s Structured Outputs and function‑calling mode guarantee that the model’s reply conforms to any JSON Schema you supply, so a renderer or agent can trust the format ([OpenAI Platform][2], [OpenAI Platform][3]).
For visual hints, a parallel **uiSchema**—popularised by *react‑jsonschema‑form*—decorates each node with layout choices without polluting the logical tree ([rjsf-team.github.io][4]).
Libraries such as Builder.io’s **Mitosis** compile that same JSON into React, Vue, Solid, or native‑mobile code, which keeps your generation pipeline framework‑agnostic ([mitosis.builder.io][5], [Just Some Dev][6]).

---

## 2 Defining a minimal component schema

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

The `enum` keyword pins every node to a finite set of component types, which the model cannot misspell or invent ([JSON Schema][7]).  Because each child item `$ref`s the root schema, the tree may nest arbitrarily deep while staying valid.

---

## 3 Live session: describing, editing, and acting on the UI

### 3.1 Inspect the document

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

`genui.describe()` returns the exact tree, so a downstream agent can reason over IDs, roles, or text content before deciding what to do.

### 3.2 Update the heading style

```python
# Promote the event title to Heading 2
genui.update(element_id="event_title", style={"variant": "heading2"})
```

```text
✓ Style modification succeeded
```

Behind the scenes, the platform patches the JSON tree, then re‑renders the view—no brittle DOM queries required.

### 3.3 Trigger a primary CTA

```python
genui.action(click_button="get_started_btn")
```

```text
✓ Button click succeeded
```

An action call maps cleanly to an analytics event or navigation hook, closing the loop between model output and user interaction.

---

## 4 Conclusion & next steps

Generative UI is no longer science fiction: with a well‑scoped JSON Schema, schema‑aware decoding, and a thin execution layer (`genui` in our demo), you can let LLMs author and edit live interfaces while keeping every change type‑safe, diff‑able, and framework‑agnostic.  From here, experiment with:

* Auto‑generating **forms** directly from a domain schema (see the rjsf uiSchema pattern) ([rjsf-team.github.io][14]).
* Compiling your JSON tree to multiple runtimes using Mitosis ([Just Some Dev][6]).
* Logging `genui.action` events to feed a reinforcement‑learning loop that refines your prompts over time.

With structure and validation in place, your model can focus on creativity—and your front‑end stays robust.

[1]: https://medium.com/%40mehdi-zare/generative-ui-building-dynamic-interfaces-with-llms-and-ai-b515d943b9aa?utm_source=chatgpt.com "Generative UI: Building Dynamic Interfaces with LLMs and AI - Medium"
[2]: https://platform.openai.com/docs/guides/structured-outputs?utm_source=chatgpt.com "Structured Outputs - OpenAI API"
[3]: https://platform.openai.com/docs/guides/function-calling?utm_source=chatgpt.com "Function calling - OpenAI API"
[4]: https://rjsf-team.github.io/react-jsonschema-form/docs/api-reference/uiSchema/?utm_source=chatgpt.com "uiSchema | react-jsonschema-form - GitHub Pages"
[5]: https://mitosis.builder.io/docs/customizability/?utm_source=chatgpt.com "Customization - Mitosis - Builder.io"
[6]: https://www.nickyt.co/blog/build-framework-agnostic-components-with-mitosis-4c4k?utm_source=chatgpt.com "Build framework-agnostic components with Mitosis - Just Some Dev"
[7]: https://json-schema.org/understanding-json-schema/reference/enum?utm_source=chatgpt.com "Enumerated values - JSON Schema"
[8]: https://arxiv.org/html/2403.06988?utm_source=chatgpt.com "Guiding LLMs The Right Way: Fast, Non-Invasive Constrained ..."
[9]: https://github.com/Saibo-creator/Awesome-LLM-Constrained-Decoding?utm_source=chatgpt.com "Saibo-creator/Awesome-LLM-Constrained-Decoding - GitHub"
[10]: https://openai.com/index/introducing-structured-outputs-in-the-api/?utm_source=chatgpt.com "Introducing Structured Outputs in the API - OpenAI"
[11]: https://ainoya.dev/posts/llm-json-output-format-gen/?utm_source=chatgpt.com "Developing a Web UI for Controlling LLM JSON Output - ainoya.dev"
[12]: https://www.builder.io/blog/mitosis-a-quick-guide?utm_source=chatgpt.com "A Quick Guide to Mitosis: Why You Need It and How You Can Use It"
[13]: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout/Relationship_of_grid_layout_with_other_layout_methods?utm_source=chatgpt.com "Relationship of grid layout to other layout methods - CSS"
[14]: https://rjsf-team.github.io/react-jsonschema-form/docs/?utm_source=chatgpt.com "Introduction | react-jsonschema-form - GitHub Pages"

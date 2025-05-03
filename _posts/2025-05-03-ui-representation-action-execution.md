# UI representation and action execution for Generative UI

## 1 Structured UI Representation

Large models excel at generating structured UI representations. Instead of verbose HTML/JSX, we use a compact JSON component tree that's easy to parse and validate. This approach offers several advantages:

- **Type Safety**: Using JSON Schema ensures valid component structures
- **Framework Agnostic**: The same representation can be compiled to different frameworks
- **Efficient Parsing**: Compact structure reduces token usage and processing overhead

OpenAI's Structured Outputs and function-calling mode guarantee that the model's responses conform to our schema, making the UI generation process reliable and predictable.

---

## 2 Component Schema and UI Structure

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

## 3 UI Action and Interaction

### 3.1 Inspecting UI State

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

### 3.2 Modifying UI Elements

```python
# Update component properties
genui.update(element_id="event_title", style={"variant": "heading2"})
```

```text
✓ Style modification succeeded
```

Updates are applied directly to the component tree, ensuring type safety and predictable rendering.

### 3.3 Handling User Actions

```python
genui.action(click_button="get_started_btn")
```

```text
✓ Button click succeeded
```

Action handlers map user interactions to specific behaviors, creating a clear connection between UI elements and their functionality.

---

## 4 Implementation and Best Practices

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
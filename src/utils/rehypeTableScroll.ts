import type { Root, Element } from "hast";

/** Keep table semantics while containing wide research tables on small screens. */
export default function rehypeTableScroll() {
  return (tree: Root) => {
    function visit(parent: Root | Element) {
      parent.children = parent.children.map((child) => {
        if (child.type !== "element") return child;
        if (child.tagName === "table") {
          return {
            type: "element",
            tagName: "div",
            properties: {
              className: ["table-scroll"],
              tabIndex: 0,
              role: "region",
              ariaLabel: "Scrollable data table",
            },
            children: [child],
          } satisfies Element;
        }
        visit(child);
        return child;
      });
    }
    visit(tree);
  };
}

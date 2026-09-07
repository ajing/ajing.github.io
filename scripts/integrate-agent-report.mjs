/** Adapt the validated portable reader to the blog shell; chart code/data stay intact. */
import fs from "node:fs";
import zlib from "node:zlib";
import postcss from "postcss";
import selectorParser from "postcss-selector-parser";
import { parse, serialize } from "parse5";
const base = "public/reports/multilingual-agent-language-gaps/";
const html = fs.readFileSync(base + "index.html", "utf8");
const encoded = html.match(
  /<template id="data-analytics-portable-reader-runtime-source"[^>]*>([\s\S]*?)<\/template>/,
)[1];
const runtime = zlib.gunzipSync(Buffer.from(encoded, "base64")).toString();
let script = runtime.match(/<script[^>]*>([\s\S]*?)<\/script>/)[1];
// Only the mount, semantic shell, and portal destination differ from the portable reader.
const boot = script.indexOf(
  "var Rq=document.getElementById(`data-analytics-portable-reader-root`)",
);
if (boot < 0) throw Error("Unknown reader bootstrap");
script = script.slice(0, boot);
const shell =
  "function CG({children:e,detailMode:t=!1,isEditMode:n=!1,surface:r}){return(0,B.jsx)(`main`";
if (!script.includes(shell)) throw Error("Unknown reader shell");
script = script.replace(shell, shell.replace("`main`", "`div`"));
if ((script.match(/\),document\.body\)/g) || []).length !== 2)
  throw Error("Unknown portal destinations");
script = script.replaceAll(
  "),document.body)",
  '),document.getElementById("integrated-agent-report"))',
);
script +=
  '\nexport function mountArtifact(root, artifact, onReady) { const instance=(0,le.createRoot)(root); instance.render((0,B.jsx)(Iq,{artifact,displayMode:"fullscreen",environment:MW,onReady})); return ()=>instance.unmount(); }\n';
fs.writeFileSync(base + "native-reader.js", script);
function scope(css) {
  const root = postcss.parse(css);
  root.walkRules((rule) => {
    if (rule.parent.type === "atrule" && /keyframes$/.test(rule.parent.name))
      return;
    rule.selector = selectorParser((selectors) => {
      selectors.each((sel) => {
        let anchored = false;
        sel.walkPseudos((n) => {
          if (n.value === ":root") {
            n.replaceWith(
              selectorParser.id({ value: "integrated-agent-report" }),
            );
            anchored = true;
          }
        });
        sel.walkTags((n) => {
          if (["html", "body"].includes(n.value)) {
            n.replaceWith(
              selectorParser.id({ value: "integrated-agent-report" }),
            );
            anchored = true;
          }
        });
        sel.walkIds((n) => {
          if (n.value === "root") {
            n.value = "integrated-agent-report";
            anchored = true;
          }
        });
        if (!anchored) {
          const first = sel.nodes[0];
          if (first?.type === "attribute" && first.attribute === "data-theme")
            sel.prepend(
              selectorParser.id({ value: "integrated-agent-report" }),
            );
          else {
            sel.prepend(selectorParser.combinator({ value: " " }));
            sel.prepend(
              selectorParser.id({ value: "integrated-agent-report" }),
            );
          }
        }
      });
    }).processSync(rule.selector);
  });
  return root.toString();
}
const css = [...runtime.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)]
  .map((m) => m[1])
  .join("\n");
const fallbackCss = html.match(
  /<style data-data-analytics-portable-fallback="true">([\s\S]*?)<\/style>/,
)[1];
fs.writeFileSync(base + "native-reader.css", scope(fallbackCss + "\n" + css));
const doc = parse(html);
function find(n, p) {
  if (p(n)) return n;
  for (const c of n.childNodes || []) {
    const r = find(c, p);
    if (r) return r;
  }
}
function attr(n, k) {
  return n.attrs?.find((a) => a.name === k)?.value;
}
const fallback = find(
  doc,
  (n) => attr(n, "id") === "data-analytics-portable-fallback",
);
function remove(n, p) {
  n.childNodes = (n.childNodes || []).filter((c) => !p(c));
  for (const c of n.childNodes) remove(c, p);
}
remove(
  fallback,
  (n) =>
    (n.tagName === "header" && attr(n, "class") === "portable-page-header") ||
    attr(n, "data-artifact-block-id") === "text-0",
);
fs.writeFileSync(
  "src/generated/agent-report/fallback.html",
  serialize(fallback),
);
const a = JSON.parse(fs.readFileSync(base + "artifact.json", "utf8"));
fs.writeFileSync(
  "src/generated/agent-report/content.json",
  JSON.stringify({
    title: a.manifest.title,
    charts: a.manifest.charts.length,
    tables: a.manifest.tables.length,
    blocks: a.manifest.blocks.length,
  }),
);
console.log(
  "Native reader prepared: same 15 charts, 5 tables; one blog title; no frame.",
);

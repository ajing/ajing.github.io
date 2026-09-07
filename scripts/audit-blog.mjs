import fs from "node:fs";
import path from "node:path";
import { parse } from "parse5";
const root = process.cwd(),
  dir = path.join(root, "dist/posts");
const reports = [];
function walk(n, f) {
  f(n);
  for (const c of n.childNodes || []) walk(c, f);
}
function attr(n, k) {
  return n.attrs?.find((a) => a.name === k)?.value;
}
for (const slug of fs.readdirSync(dir)) {
  const file = path.join(dir, slug, "index.html");
  if (!fs.existsSync(file)) continue;
  const tree = parse(fs.readFileSync(file, "utf8"));
  let article;
  const ids = new Set();
  walk(tree, (n) => {
    if (attr(n, "id")) ids.add(attr(n, "id"));
    if (attr(n, "id") === "article") article = n;
  });
  if (!article) continue;
  const issues = [];
  let tables = 0,
    maths = 0;
  walk(article, (n) => {
    if (n.tagName === "table") tables++;
    if ((attr(n, "class") || "").split(" ").includes("katex")) maths++;
    if ((attr(n, "class") || "").includes("katex-error"))
      issues.push("math-error: " + attr(n, "title"));
    if (n.tagName === "img" && !attr(n, "alt"))
      issues.push("missing-alt: " + attr(n, "src"));
    if (n.tagName === "h1") issues.push("body-h1");
    const url =
      n.tagName === "a"
        ? attr(n, "href")
        : n.tagName === "img"
          ? attr(n, "src")
          : null;
    if (!url || /^(https?:|mailto:|data:|tel:)/.test(url)) return;
    if (url.startsWith("#")) {
      if (url !== "#" && !ids.has(decodeURIComponent(url.slice(1))))
        issues.push("missing-anchor: " + url);
      return;
    }
    const dest = path
      .resolve(
        url.startsWith("/") ? path.join(root, "dist") : path.dirname(file),
        "." + (url.startsWith("/") ? url : "/" + url),
      )
      .split("#")[0]
      .split("?")[0];
    if (!fs.existsSync(dest) && !fs.existsSync(path.join(dest, "index.html")))
      issues.push("missing-local: " + url);
  });
  reports.push({ slug, tables, maths, issues: [...new Set(issues)] });
}
console.log(
  JSON.stringify(
    {
      posts: reports.length,
      tables: reports.reduce((n, r) => n + r.tables, 0),
      mathExpressions: reports.reduce((n, r) => n + r.maths, 0),
      reports,
    },
    null,
    2,
  ),
);
process.exitCode = reports.some((r) => r.issues.length) ? 1 : 0;

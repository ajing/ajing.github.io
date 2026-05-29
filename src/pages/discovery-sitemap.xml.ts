import type { APIRoute } from "astro";

const discoveryPaths = [
  "rss.xml",
  "feed.json",
  "llms.txt",
  "llms-full.txt",
] as const;

export const GET: APIRoute = ({ site }) => {
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${discoveryPaths
    .map(path => `  <url>\n    <loc>${new URL(path, site).href}</loc>\n  </url>`)
    .join("\n")}\n</urlset>\n`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/xml",
    },
  });
};


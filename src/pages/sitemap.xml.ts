import type { APIRoute } from "astro";

export const GET: APIRoute = ({ site }) => {
  const sitemapURLs = ["sitemap-0.xml", "discovery-sitemap.xml"].map(path =>
    new URL(path, site)
  );
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${sitemapURLs
    .map(url => `  <sitemap>\n    <loc>${url.href}</loc>\n  </sitemap>`)
    .join("\n")}\n</sitemapindex>\n`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/xml",
    },
  });
};

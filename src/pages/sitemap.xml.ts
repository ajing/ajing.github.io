import type { APIRoute } from "astro";

export const GET: APIRoute = ({ site }) => {
  const sitemapURL = new URL("sitemap-0.xml", site);
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <sitemap>\n    <loc>${sitemapURL.href}</loc>\n  </sitemap>\n</sitemapindex>\n`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/xml",
    },
  });
};

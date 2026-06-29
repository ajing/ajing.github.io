import type { APIRoute } from "astro";
import { SITE } from "@/config";

export const GET: APIRoute = ({ site }) => {
  const origin = site ?? new URL(SITE.website);
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <sitemap>\n    <loc>${new URL("sitemap.xml", origin).href}</loc>\n  </sitemap>\n</sitemapindex>\n`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
    },
  });
};

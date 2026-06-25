import type { APIRoute } from "astro";
import { getCanonicalSitemapEntries } from "@/utils/sitemapUrls";

const escapeXml = (value: string) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const urlEntry = ({ loc, lastmod }: { loc: string; lastmod?: string }) =>
  [
    "  <url>",
    `    <loc>${escapeXml(loc)}</loc>`,
    lastmod && `    <lastmod>${lastmod}</lastmod>`,
    "  </url>",
  ]
    .filter(Boolean)
    .join("\n");

export const GET: APIRoute = async ({ site }) => {
  const urls = await getCanonicalSitemapEntries(site);

  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls
    .map(urlEntry)
    .join("\n")}\n</urlset>\n`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
    },
  });
};

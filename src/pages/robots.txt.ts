import type { APIRoute } from "astro";
import { SITE } from "@/config";

const getRobotsTxt = (sitemapURL: URL) => `
User-agent: *
Disallow:

Sitemap: ${sitemapURL.href}
`;

export const GET: APIRoute = ({ site }) => {
  const origin = site ?? new URL(SITE.website);
  return new Response(getRobotsTxt(new URL("sitemap.xml", origin)));
};

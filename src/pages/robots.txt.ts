import type { APIRoute } from "astro";
import { SITE } from "@/config";

const getRobotsTxt = (sitemapURLs: URL[]) => `
User-agent: *
Disallow:

${sitemapURLs.map((sitemapURL) => `Sitemap: ${sitemapURL.href}`).join("\n")}
`;

export const GET: APIRoute = ({ site }) => {
  const origin = site ?? new URL(SITE.website);
  return new Response(
    getRobotsTxt([
      new URL("sitemap.xml", origin),
      new URL("priority-sitemap.xml", origin),
      new URL("index.xml", origin),
    ]),
  );
};

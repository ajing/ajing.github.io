import type { APIRoute } from "astro";
import { getCanonicalSitemapEntries } from "@/utils/sitemapUrls";

export const GET: APIRoute = async ({ site }) => {
  const urls = await getCanonicalSitemapEntries(site);
  const body = `${urls.map(({ loc }) => loc).join("\n")}\n`;

  return new Response(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
};

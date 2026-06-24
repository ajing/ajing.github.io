import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { SITE } from "@/config";
import { TOPICS } from "@/data/topics";
import { getPath } from "@/utils/getPath";
import getSortedPosts from "@/utils/getSortedPosts";

const staticPages = [
  "/",
  "/about/",
  "/archives/",
  "/posts/",
  "/start-here/",
  "/topics/",
] as const;

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
  const origin = site ?? new URL(SITE.website);
  const posts = getSortedPosts(await getCollection("blog"));
  const urls = [
    ...staticPages.map((path) => ({ loc: new URL(path, origin).href })),
    ...TOPICS.map((topic) => ({
      loc: new URL(`/topics/${topic.slug}/`, origin).href,
    })),
    ...posts.map((post) => ({
      loc: new URL(`${getPath(post.id, post.filePath)}/`, origin).href,
      lastmod: (post.data.modDatetime ?? post.data.pubDatetime)
        .toISOString()
        .slice(0, 10),
    })),
  ];

  const dedupedUrls = Array.from(
    new Map(urls.map((url) => [url.loc, url])).values(),
  );

  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${dedupedUrls
    .map(urlEntry)
    .join("\n")}\n</urlset>\n`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
    },
  });
};

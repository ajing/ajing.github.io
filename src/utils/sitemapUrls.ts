import { getCollection } from "astro:content";
import { SITE } from "@/config";
import { TOPICS } from "@/data/topics";
import { getPath } from "@/utils/getPath";
import getSortedPosts from "@/utils/getSortedPosts";

export interface SitemapEntry {
  loc: string;
  lastmod?: string;
}

const staticPages = [
  "/",
  "/about/",
  "/archives/",
  "/posts/",
  "/start-here/",
  "/topics/",
] as const;

export const getCanonicalSitemapEntries = async (
  site?: URL,
): Promise<SitemapEntry[]> => {
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

  return Array.from(new Map(urls.map((url) => [url.loc, url])).values());
};

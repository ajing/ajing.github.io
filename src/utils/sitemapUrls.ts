import { getCollection } from "astro:content";
import { SITE } from "@/config";
import { TOPICS } from "@/data/topics";
import { getPath } from "@/utils/getPath";
import getSortedPosts from "@/utils/getSortedPosts";

export interface SitemapEntry {
  loc: string;
  lastmod: string;
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
  const latestContentLastmod =
    posts[0]?.data.modDatetime ?? posts[0]?.data.pubDatetime ?? new Date();
  const latestContentLastmodText = latestContentLastmod
    .toISOString()
    .slice(0, 10);
  const postLastmod = (post: (typeof posts)[number]) =>
    (post.data.modDatetime ?? post.data.pubDatetime).toISOString().slice(0, 10);
  const topicLastmod = (topic: (typeof TOPICS)[number]) => {
    const latestTopicPost = posts.find((post) =>
      post.data.tags.some((tag) => topic.tags.includes(tag)),
    );
    return latestTopicPost
      ? postLastmod(latestTopicPost)
      : latestContentLastmodText;
  };
  const urls = [
    ...staticPages.map((path) => ({
      loc: new URL(path, origin).href,
      lastmod: latestContentLastmodText,
    })),
    ...TOPICS.map((topic) => ({
      loc: new URL(`/topics/${topic.slug}/`, origin).href,
      lastmod: topicLastmod(topic),
    })),
    ...posts.map((post) => ({
      loc: new URL(`${getPath(post.id, post.filePath)}/`, origin).href,
      lastmod: postLastmod(post),
    })),
  ];

  return Array.from(new Map(urls.map((url) => [url.loc, url])).values());
};

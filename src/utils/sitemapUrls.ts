import { getCollection } from "astro:content";
import { SITE } from "@/config";
import { TOPICS } from "@/data/topics";
import { getPath } from "@/utils/getPath";
import getPostsByTag from "@/utils/getPostsByTag";
import getSortedPosts from "@/utils/getSortedPosts";
import getUniqueTags from "@/utils/getUniqueTags";

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
  "/tags/",
  "/topics/",
] as const;

const getPaginatedPaths = (basePath: string, itemCount: number) =>
  Array.from(
    { length: Math.max(Math.ceil(itemCount / SITE.postPerPage) - 1, 0) },
    (_, index) => `${basePath}${index + 2}/`,
  );

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
  const tags = getUniqueTags(posts);
  const urls = [
    ...staticPages.map((path) => ({
      loc: new URL(path, origin).href,
      lastmod: latestContentLastmodText,
    })),
    ...getPaginatedPaths("/posts/", posts.length).map((path) => ({
      loc: new URL(path, origin).href,
      lastmod: latestContentLastmodText,
    })),
    ...tags.flatMap(({ tag }) => {
      const tagPosts = getPostsByTag(posts, tag);
      const tagLastmod = tagPosts[0]
        ? postLastmod(tagPosts[0])
        : latestContentLastmodText;

      return [
        {
          loc: new URL(`/tags/${tag}/`, origin).href,
          lastmod: tagLastmod,
        },
        ...getPaginatedPaths(`/tags/${tag}/`, tagPosts.length).map((path) => ({
          loc: new URL(path, origin).href,
          lastmod: tagLastmod,
        })),
      ];
    }),
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

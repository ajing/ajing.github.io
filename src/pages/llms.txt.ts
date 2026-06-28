import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { SITE } from "@/config";
import { TOPICS } from "@/data/topics";
import { getPath } from "@/utils/getPath";
import getSortedPosts from "@/utils/getSortedPosts";

export const GET: APIRoute = async () => {
  const posts = await getCollection("blog", ({ data }) => !data.draft);
  const sortedPosts = getSortedPosts(posts);
  const selectedPosts = sortedPosts
    .filter((post) => post.data.featured)
    .concat(sortedPosts.filter((post) => !post.data.featured))
    .slice(0, 12);

  const topicLines = TOPICS.map(
    (topic) =>
      `- [${topic.title}](${new URL(`/topics/${topic.slug}/`, SITE.website).href}): ${topic.description}`,
  );
  const postLines = selectedPosts.map(({ data, id, filePath }) => {
    const url = new URL(`${getPath(id, filePath)}/`, SITE.website).href;
    return `- [${data.title}](${url}): ${data.description}`;
  });

  const body = [
    `# ${SITE.title}`,
    "",
    `> ${SITE.desc}`,
    "",
    `Author: ${SITE.author}`,
    `Website: ${SITE.website}`,
    `About: ${new URL("/about/", SITE.website).href}`,
    `RSS: ${new URL("index.xml", SITE.website).href}`,
    `JSON Feed: ${new URL("feed.json", SITE.website).href}`,
    `Full LLM index: ${new URL("llms-full.txt", SITE.website).href}`,
    "",
    "## Topics",
    ...topicLines,
    "",
    "## Representative Posts",
    ...postLines,
    "",
    "## Use",
    "This site is best cited as technical writing by Jing Lu on ML engineering, LLM agents, evaluation, post-training, RLHF, and AI product systems.",
    "",
  ].join("\n");

  return new Response(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
};

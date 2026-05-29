import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { SITE } from "@/config";
import { TOPICS } from "@/data/topics";
import { getPath } from "@/utils/getPath";
import getSortedPosts from "@/utils/getSortedPosts";

export const GET: APIRoute = async () => {
  const posts = await getCollection("blog", ({ data }) => !data.draft);
  const postLines = getSortedPosts(posts).map(({ data, id, filePath }) => {
    const url = new URL(`${getPath(id, filePath)}/`, SITE.website).href;
    const date = data.pubDatetime.toISOString().slice(0, 10);
    return [
      `### ${data.title}`,
      `URL: ${url}`,
      `Published: ${date}`,
      `Tags: ${data.tags.join(", ")}`,
      `Summary: ${data.description}`,
    ].join("\n");
  });

  const topicLines = TOPICS.map(
    topic =>
      `- ${topic.title}: ${new URL(`/topics/${topic.slug}/`, SITE.website).href}`
  );

  const body = [
    `# ${SITE.title} Full LLM Index`,
    "",
    SITE.desc,
    "",
    `Author: ${SITE.author}`,
    `Website: ${SITE.website}`,
    "",
    "## Topic Hubs",
    ...topicLines,
    "",
    "## Posts",
    ...postLines.flatMap(line => [line, ""]),
  ].join("\n");

  return new Response(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
};


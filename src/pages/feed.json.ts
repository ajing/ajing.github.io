import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { SITE } from "@/config";
import { getPath } from "@/utils/getPath";
import getSortedPosts from "@/utils/getSortedPosts";

export const GET: APIRoute = async () => {
  const posts = await getCollection("blog", ({ data }) => !data.draft);
  const items = getSortedPosts(posts).map(({ data, id, filePath }) => {
    const url = new URL(`${getPath(id, filePath)}/`, SITE.website).href;

    return {
      id: url,
      url,
      title: data.title,
      summary: data.description,
      content_text: data.description,
      date_published: data.pubDatetime.toISOString(),
      ...(data.modDatetime && {
        date_modified: data.modDatetime.toISOString(),
      }),
      authors: [{ name: data.author ?? SITE.author }],
      tags: data.tags,
    };
  });

  return new Response(
    JSON.stringify(
      {
        version: "https://jsonfeed.org/version/1.1",
        title: SITE.title,
        home_page_url: SITE.website,
        feed_url: new URL("feed.json", SITE.website).href,
        description: SITE.desc,
        authors: [{ name: SITE.author, url: SITE.profile }],
        language: SITE.lang,
        items,
      },
      null,
      2
    ),
    {
      headers: {
        "Content-Type": "application/feed+json; charset=utf-8",
      },
    }
  );
};


import { buildRssFeed } from "@/utils/rssFeed";

export async function GET() {
  return buildRssFeed();
}

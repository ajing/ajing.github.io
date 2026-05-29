import { SITE } from "@/config";
import { SOCIALS } from "@/constants";

export const sameAsLinks = SOCIALS.map(social => social.href);

export function personStructuredData() {
  return {
    "@context": "https://schema.org",
    "@type": "Person",
    name: SITE.author,
    alternateName: "ajing",
    url: SITE.profile,
    sameAs: sameAsLinks,
    knowsAbout: [
      "Machine Learning",
      "Large Language Models",
      "AI Agents",
      "RLHF",
      "Post-Training",
      "Agent Evaluation",
    ],
  };
}

export function websiteStructuredData(url: string) {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: SITE.title,
    url,
    description: SITE.desc,
    author: personStructuredData(),
  };
}

export function breadcrumbStructuredData(items: { name: string; url: string }[]) {
  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: items.map((item, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: item.name,
      item: item.url,
    })),
  };
}

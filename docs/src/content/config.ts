import { defineCollection } from "astro:content";
import { docsSchema } from "@astrojs/starlight/schema";

// Declares the `docs` content collection Starlight renders from. Required —
// without it Astro finds no entries and the build reports "No pages found".
export const collections = {
  docs: defineCollection({ schema: docsSchema() }),
};

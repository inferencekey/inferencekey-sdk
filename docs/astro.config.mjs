// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// Static documentation site for the InferenceKey SDK.
// Starlight ships Pagefind search, dark mode, copy buttons, SEO (sitemap +
// Open Graph) and i18n out of the box. EN is the source locale; ES/FR fall
// back to EN per page until translated (Starlight default), so launch is never
// blocked on full translation coverage.
export default defineConfig({
  site: "https://docs.inferencekey.com",
  // Default locale lives under /en/; send the bare root there.
  redirects: {
    "/": "/en/",
  },
  integrations: [
    starlight({
      title: "InferenceKey SDK",
      description:
        "Declare AI workloads in code, ensure they exist on the platform, and call the resulting OpenAI-compatible endpoints.",
      // Brand: the InferenceKey wordmark as the top-bar logo (replacing the
      // "InferenceKey SDK" text), per-theme so it reads on dark and light.
      // The iK monogram stays as the favicon. Assets come from docs/brand/logos.
      logo: {
        dark: "./src/assets/brand/wordmark-dark.svg",
        light: "./src/assets/brand/wordmark-light.svg",
        alt: "InferenceKey",
        replacesTitle: true,
      },
      favicon: "/favicon.svg",
      customCss: ["./src/styles/brand.css"],
      head: [
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap",
          },
        },
        // Open Graph / social preview image (brand asset).
        {
          tag: "meta",
          attrs: { property: "og:image", content: "https://docs.inferencekey.com/og.png" },
        },
        {
          tag: "meta",
          attrs: { name: "twitter:image", content: "https://docs.inferencekey.com/og.png" },
        },
        {
          tag: "meta",
          attrs: { name: "twitter:card", content: "summary_large_image" },
        },
      ],
      defaultLocale: "en",
      locales: {
        en: { label: "English", lang: "en" },
        es: { label: "Español", lang: "es" },
        fr: { label: "Français", lang: "fr" },
      },
      social: {
        github: "https://github.com/inferencekey/inferencekey-sdk",
      },
      editLink: {
        baseUrl:
          "https://github.com/inferencekey/inferencekey-sdk/edit/main/docs/",
      },
      // Cross-link back to the Manager dashboard and the landing (plan C32).
      // Footer/landing links live in the page content; the dashboard CTA is a
      // top-level nav link here.
      sidebar: [
        {
          label: "Start here",
          translations: { es: "Empieza aquí", fr: "Commencer" },
          autogenerate: { directory: "quickstart" },
        },
        {
          label: "Guides",
          translations: { es: "Guías", fr: "Guides" },
          autogenerate: { directory: "guides" },
        },
        {
          label: "Reference",
          translations: { es: "Referencia", fr: "Référence" },
          autogenerate: { directory: "reference" },
        },
        {
          label: "API reference",
          translations: { es: "Referencia de API", fr: "Référence API" },
          autogenerate: { directory: "api" },
        },
      ],
    }),
  ],
});

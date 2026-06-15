# Build the static Starlight site, then serve it with nginx.
# Mirrors the landing's Docker→registry→Swarm pattern; the only runtime is a
# tiny static file server (no SSR), per the docs plan (C27).

# --- build stage ---
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
# Use npm install (no committed lockfile for the docs project yet); CI may
# switch to `npm ci` once a lockfile is added.
RUN npm install
COPY . .
RUN npm run docs:build

# --- serve stage ---
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1
CMD ["nginx", "-g", "daemon off;"]

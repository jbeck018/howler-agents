# Multi-stage build for Howler Agents Docs
FROM node:22-slim AS builder

WORKDIR /app

# Install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy workspace root config
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml* ./

# Copy docs package
COPY packages/howler-agents-docs packages/howler-agents-docs

# Install dependencies
RUN pnpm install --frozen-lockfile || pnpm install

# Build the docs site
RUN cd packages/howler-agents-docs && pnpm run build

# Runtime — serve with nginx
FROM nginx:alpine AS runtime

COPY --from=builder /app/packages/howler-agents-docs/dist /usr/share/nginx/html

# SPA routing: serve files under /docs-site/ prefix (matching Vite base)
RUN printf 'server {\n\
    listen 3001;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location /docs-site/ {\n\
        alias /usr/share/nginx/html/;\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
    location /docs-site {\n\
        return 301 /docs-site/;\n\
    }\n\
    location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff2?)$ {\n\
        expires 1y;\n\
        add_header Cache-Control "public, immutable";\n\
    }\n\
    location = / {\n\
        return 301 /docs-site/;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 3001

CMD ["nginx", "-g", "daemon off;"]

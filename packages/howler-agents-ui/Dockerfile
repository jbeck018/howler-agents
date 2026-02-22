# Multi-stage build for Howler Agents UI
FROM node:22-slim AS builder

WORKDIR /app

# Install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy workspace root config
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml* ./

# Copy UI and SDK packages (UI depends on SDK types)
COPY packages/howler-agents-ui packages/howler-agents-ui
COPY packages/howler-agents-ts packages/howler-agents-ts

# Install dependencies
RUN pnpm install --frozen-lockfile || pnpm install

# Build the UI
RUN cd packages/howler-agents-ui && pnpm run build

# Runtime — serve with nginx
FROM nginx:alpine AS runtime

COPY --from=builder /app/packages/howler-agents-ui/dist /usr/share/nginx/html

# SPA routing: serve index.html for all routes
RUN printf 'server {\n\
    listen 3000;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
    location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff2?)$ {\n\
        expires 1y;\n\
        add_header Cache-Control "public, immutable";\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]

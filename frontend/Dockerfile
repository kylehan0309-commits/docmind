# DocMind frontend: build the Vite bundle, then serve it from nginx which also
# reverse-proxies /api to the backend (mirrors the Vite dev proxy).
FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
# nginx's stock entrypoint runs envsubst over every *.template here on start,
# writing the result to conf.d/ - only vars that are actually set in the
# environment get substituted, so nginx's own $host/$remote_addr etc. are safe.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

# Defaults match the docker-compose network (the backend service is reachable
# as plain "backend" there, via Docker's embedded DNS at 127.0.0.11). Fly.io
# overrides these three in each app's fly.toml [env].
ENV BACKEND_HOST=backend \
    BACKEND_PORT=8000 \
    DNS_RESOLVER=127.0.0.11

EXPOSE 80

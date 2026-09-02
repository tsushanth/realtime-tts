# Plain Node app, no native deps — nothing fancier than this needed.
FROM node:22-slim

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY . .

ENV PORT=8090
EXPOSE 8090

CMD ["node", "server.js"]

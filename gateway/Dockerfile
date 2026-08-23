FROM node:22-alpine
WORKDIR /app
COPY package.json .
RUN npm install --omit=dev
COPY server.js runpod-adapter.js pod-manager.js keys.js .
ENV PORT=8080
EXPOSE 8080
CMD ["node", "server.js"]

FROM node:22-alpine
# ffmpeg encodes mp3/opus output for POST /v1/text-to-speech (see audioEncode.js)
RUN apk add --no-cache ffmpeg
WORKDIR /app
COPY package.json .
RUN npm install --omit=dev
COPY server.js runpod-adapter.js keys.js voiceApiProxy.js audit.js audioEncode.js voices.js orpheusApiProxy.js .
ENV PORT=8080
EXPOSE 8080
CMD ["node", "server.js"]

// Realtime TTS gateway: terminates client WebSockets on Fly, proxies each session 1:1 to
// a GPU inference worker (RunPod, or the local worker/server.py during dev). Kept
// deliberately thin — no batching, no session multiplexing — so it adds near-zero
// latency on top of the worker's own TTFB. See ../DECISIONS.md.
import { WebSocketServer, WebSocket } from "ws";
import http from "node:http";
import { runpodConfigured, handleClientOverRunpod } from "./runpod-adapter.js";

const PORT = process.env.PORT || 8080;
const WORKER_URL = process.env.WORKER_WS_URL || "ws://127.0.0.1:8765";
const USE_RUNPOD = runpodConfigured();

const server = http.createServer((req, res) => {
  if (req.url === "/health") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({
      status: "ok",
      backend: USE_RUNPOD ? "runpod" : "direct-ws",
      worker: USE_RUNPOD ? process.env.RUNPOD_ENDPOINT_ID : WORKER_URL,
    }));
    return;
  }
  res.writeHead(404);
  res.end();
});

const wss = new WebSocketServer({ server, path: "/tts" });

wss.on("connection", (client) => {
  if (USE_RUNPOD) {
    handleClientOverRunpod(client);
    return;
  }
  const worker = new WebSocket(WORKER_URL);
  let workerOpen = false;
  const pending = [];

  worker.on("open", () => {
    workerOpen = true;
    for (const frame of pending.splice(0)) worker.send(frame);
  });

  worker.on("message", (data, isBinary) => {
    if (client.readyState === WebSocket.OPEN) client.send(data, { binary: isBinary });
  });

  worker.on("error", (err) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ type: "error", message: `worker: ${err.message}` }));
    }
  });

  worker.on("close", () => {
    if (client.readyState === WebSocket.OPEN) client.close();
  });

  client.on("message", (data, isBinary) => {
    if (workerOpen) worker.send(data, { binary: isBinary });
    else pending.push(data);
  });

  client.on("close", () => {
    if (worker.readyState === WebSocket.OPEN || worker.readyState === WebSocket.CONNECTING) {
      worker.close();
    }
  });

  client.on("error", () => worker.close());
});

server.listen(PORT, () => {
  const target = USE_RUNPOD ? `runpod:${process.env.RUNPOD_ENDPOINT_ID}` : WORKER_URL;
  console.log(`gateway listening on :${PORT}, backend=${target}`);
});

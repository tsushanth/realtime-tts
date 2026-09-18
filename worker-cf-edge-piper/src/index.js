// Anycast edge-termination proxy in front of the Modal TTS worker.
//
// Why this exists: Modal's *.modal.run endpoints resolve to a single AWS
// region (us-east-1), so every client pays a full transcontinental round
// trip for TCP+TLS+WS-upgrade before the first audio byte — measured at
// ~110ms extra vs ElevenLabs/Replicate, which terminate TLS at an anycast
// edge near the client. This Worker runs at Cloudflare's edge (hundreds of
// PoPs), terminates the client's connection close to them, then opens its
// own outbound connection to Modal over Cloudflare's backbone — the same
// trick those providers use, without needing Cloudflare's paid Origin
// Rules SNI-override feature (not available on this zone's plan).
//
// This is a byte-transparent proxy: it does not parse the app-level
// synthesize/chunk_meta protocol, just relays frames both directions.

const MODAL_ORIGIN = "t-sushanth--realtime-tts-worker-piper-web.modal.run";

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const upgradeHeader = request.headers.get("Upgrade");

    if (!upgradeHeader || upgradeHeader.toLowerCase() !== "websocket") {
      // Non-WS requests (e.g. a stray GET /health) — proxy through as plain HTTP
      // so the route is fully transparent, not just for the WS path.
      const originUrl = new URL(url.pathname + url.search, `https://${MODAL_ORIGIN}`);
      const originRequest = new Request(originUrl, request);
      originRequest.headers.set("Host", MODAL_ORIGIN);
      return fetch(originRequest);
    }

    const webSocketPair = new WebSocketPair();
    const [client, server] = Object.values(webSocketPair);
    server.accept();
    // Binary frames (the PCM16 audio) default to Blob in event.data — Blob
    // isn't directly sendable via the other socket's send(), and relaying it
    // as-is silently stringifies to "[object Blob]", corrupting the audio.
    // arraybuffer gives raw bytes that send() forwards correctly.
    server.binaryType = "arraybuffer";

    const originUrl = new URL(url.pathname + url.search, `https://${MODAL_ORIGIN}`);
    // Minimal, explicit header set — forwarding the client's full header set
    // (including cf-* headers Cloudflare injects) caused the outbound fetch
    // itself to fail ("Network connection lost") rather than reach Modal.
    const originHeaders = new Headers();
    originHeaders.set("Upgrade", "websocket");
    const authHeader = request.headers.get("Authorization");
    if (authHeader) originHeaders.set("Authorization", authHeader);

    let originWs;
    try {
      const originResponse = await fetch(originUrl, {
        headers: originHeaders,
      });
      originWs = originResponse.webSocket;
      if (!originWs) {
        const body = await originResponse.text().catch(() => "");
        console.log(`origin rejected upgrade: status=${originResponse.status} body=${body.slice(0, 300)}`);
        server.close(1011, `origin did not accept WebSocket upgrade (status ${originResponse.status})`);
        return new Response(null, { status: 101, webSocket: client });
      }
    } catch (err) {
      server.close(1011, `failed to reach origin: ${err.message}`);
      return new Response(null, { status: 101, webSocket: client });
    }

    originWs.accept();
    originWs.binaryType = "arraybuffer";

    // Relay both directions. Binary frames (the PCM16 audio) pass through
    // unchanged — event.data is already an ArrayBuffer for binary messages,
    // a string for text (the JSON control messages), and send() accepts both.
    server.addEventListener("message", (event) => {
      try {
        originWs.send(event.data);
      } catch (err) {
        server.close(1011, "origin send failed");
      }
    });
    originWs.addEventListener("message", (event) => {
      try {
        server.send(event.data);
      } catch (err) {
        originWs.close(1011, "client send failed");
      }
    });

    server.addEventListener("close", (event) => {
      try {
        originWs.close(event.code, event.reason);
      } catch (err) {
        /* already closed */
      }
    });
    originWs.addEventListener("close", (event) => {
      try {
        server.close(event.code, event.reason);
      } catch (err) {
        /* already closed */
      }
    });

    server.addEventListener("error", () => {
      try {
        originWs.close(1011, "client error");
      } catch (err) {
        /* already closed */
      }
    });
    originWs.addEventListener("error", () => {
      try {
        server.close(1011, "origin error");
      } catch (err) {
        /* already closed */
      }
    });

    return new Response(null, { status: 101, webSocket: client });
  },
};

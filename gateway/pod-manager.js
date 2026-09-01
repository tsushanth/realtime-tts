// Manages the GPU RunPod Pod's lifecycle: provision on first real request, tear down
// after IDLE_TIMEOUT_MS of inactivity. State is a single in-process state machine —
// this gateway runs one machine (min_machines_running=1, see fly.toml) so that's safe;
// it would NOT be safe if the gateway ever ran multiple replicas without moving this
// state into something shared (Redis, a DB row) first.
const RUNPOD_KEY = process.env.RUNPOD_API_KEY;
const TEMPLATE_ID = process.env.RUNPOD_POD_TEMPLATE_ID || "r7cttxeun9";
const POD_IMAGE = "ghcr.io/tsushanth/realtime-tts-worker:pod";
const IDLE_TIMEOUT_MS = parseInt(process.env.POD_IDLE_TIMEOUT_MS || "900000", 10); // 15 min
const BOOT_POLL_INTERVAL_MS = 8000;
// Was 6 min ("observed boot times ranged ~90s-280s") — too tight. Confirmed
// live 2026-09-01: a cold RunPod host with no cached layers for this image
// took ~5.5 min just to pull+boot before the container ever came up, so the
// gateway gave up and deleted the pod right as it was finishing — a false
// "provisioning failed", not an actual failure. The 90-280s figure was
// evidently measured on hosts that already had the image cached.
const BOOT_TIMEOUT_MS = 10 * 60 * 1000;

const STATE = { OFF: "off", PROVISIONING: "provisioning", READY: "ready", ERROR: "error" };

let state = STATE.OFF;
let podId = null;
let workerUrl = null; // ws://<ip>:<port>
let lastRequestAt = 0;
let provisioningPromise = null;
let idleCheckTimer = null;

async function runpodFetch(url, opts = {}) {
  const res = await fetch(url, {
    ...opts,
    headers: { Authorization: `Bearer ${RUNPOD_KEY}`, "Content-Type": "application/json", ...(opts.headers || {}) },
  });
  if (!res.ok) throw new Error(`RunPod ${url} -> ${res.status} ${await res.text()}`);
  return res.json();
}

async function createPod() {
  const resp = await runpodFetch("https://rest.runpod.io/v1/pods", {
    method: "POST",
    body: JSON.stringify({
      name: "realtime-tts-pod",
      imageName: POD_IMAGE,
      gpuTypeIds: ["NVIDIA L4", "NVIDIA A40", "NVIDIA GeForce RTX 4090"],
      gpuCount: 1,
      containerDiskInGb: 10,
      env: { WORKER_HOST: "0.0.0.0", WORKER_PORT: "8765" },
      ports: ["8765/tcp", "22/tcp"],
      cloudType: "SECURE",
      supportPublicIp: true,
    }),
  });
  return resp.id;
}

async function pollForWorkerUrl(id) {
  const deadline = Date.now() + BOOT_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const resp = await fetch("https://api.runpod.io/graphql?api_key=" + RUNPOD_KEY, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: `query { pod(input: {podId: "${id}"}) { runtime { ports { ip isIpPublic privatePort publicPort type } } } }`,
      }),
    }).then((r) => r.json());
    const ports = resp?.data?.pod?.runtime?.ports || [];
    const match = ports.find((p) => p.type === "tcp" && p.isIpPublic && p.privatePort === 8765);
    if (match) return `ws://${match.ip}:${match.publicPort}`;
    await new Promise((r) => setTimeout(r, BOOT_POLL_INTERVAL_MS));
  }
  throw new Error("pod did not report a reachable worker URL within the boot timeout");
}

async function deletePod(id) {
  await fetch(`https://rest.runpod.io/v1/pods/${id}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${RUNPOD_KEY}` },
  }).catch(() => {}); // best-effort — don't let a delete failure wedge state
}

async function listOurPods() {
  const pods = await runpodFetch("https://rest.runpod.io/v1/pods");
  return pods.filter((p) => p.name === "realtime-tts-pod" && p.desiredStatus === "RUNNING");
}

// Runs once at process startup. In-process state (podId/workerUrl/idle watchdog) does
// NOT survive a gateway restart, but a real RunPod Pod we created before the restart
// absolutely does — and keeps billing. Without this, every deploy/crash/Fly restart
// orphans any pod that was running at the time: nothing tracks it, nothing tears it
// down, it bills forever. Confirmed live 2026-09-01: a `flyctl deploy` orphaned a pod
// mid-test, left running and billing until manually found and deleted via RunPod's own
// API. On boot, adopt exactly one existing pod (starting its idle watchdog immediately,
// so if nothing actually uses it it's torn down within IDLE_TIMEOUT_MS same as normal)
// and delete any extras as a safety net.
export async function reconcileOnStartup() {
  if (!RUNPOD_KEY) return;
  let existing;
  try {
    existing = await listOurPods();
  } catch (err) {
    console.error("pod reconciliation: failed to list pods, skipping:", err.message);
    return;
  }
  if (existing.length === 0) return;

  const [adopt, ...extras] = existing;
  for (const p of extras) {
    console.log(`pod reconciliation: deleting extra orphaned pod ${p.id}`);
    await deletePod(p.id);
  }

  try {
    console.log(`pod reconciliation: adopting existing pod ${adopt.id}`);
    podId = adopt.id;
    workerUrl = await pollForWorkerUrl(adopt.id);
    state = STATE.READY;
    // Treat it as freshly idle rather than assuming recent traffic — if nothing
    // claims it within IDLE_TIMEOUT_MS, the watchdog tears it down as normal.
    lastRequestAt = Date.now();
    startIdleWatchdog();
  } catch (err) {
    console.error(`pod reconciliation: adopted pod ${adopt.id} unreachable, deleting:`, err.message);
    await deletePod(adopt.id);
    podId = null;
    workerUrl = null;
    state = STATE.OFF;
  }
}

// Called on every authenticated request. Returns the current status immediately;
// if OFF, kicks off provisioning in the background (idempotent — concurrent callers
// share the same in-flight promise) and the caller is expected to poll/wait.
export function touchAndGetStatus() {
  lastRequestAt = Date.now();
  if (state === STATE.OFF) {
    state = STATE.PROVISIONING;
    provisioningPromise = provision();
  }
  return { state, workerUrl };
}

async function provision() {
  try {
    podId = await createPod();
    workerUrl = await pollForWorkerUrl(podId);
    state = STATE.READY;
    startIdleWatchdog();
  } catch (err) {
    state = STATE.ERROR;
    console.error("pod provisioning failed:", err.message);
    if (podId) await deletePod(podId);
    podId = null;
    workerUrl = null;
    // Allow a fresh attempt on the next request rather than sticking in ERROR forever.
    setTimeout(() => {
      if (state === STATE.ERROR) state = STATE.OFF;
    }, 30000);
  }
}

export function waitUntilReady(timeoutMs = BOOT_TIMEOUT_MS) {
  return Promise.race([
    provisioningPromise || Promise.resolve(),
    new Promise((_, reject) => setTimeout(() => reject(new Error("timed out waiting for pod")), timeoutMs)),
  ]).then(() => ({ state, workerUrl }));
}

function startIdleWatchdog() {
  if (idleCheckTimer) return;
  idleCheckTimer = setInterval(async () => {
    if (state === STATE.READY && Date.now() - lastRequestAt > IDLE_TIMEOUT_MS) {
      console.log(`pod ${podId} idle for >${IDLE_TIMEOUT_MS}ms, tearing down`);
      const id = podId;
      state = STATE.OFF;
      podId = null;
      workerUrl = null;
      clearInterval(idleCheckTimer);
      idleCheckTimer = null;
      await deletePod(id);
    }
  }, 60000);
}

export function getState() {
  return { state, podId, workerUrl, lastRequestAt, idleTimeoutMs: IDLE_TIMEOUT_MS };
}

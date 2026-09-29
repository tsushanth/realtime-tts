"""
FastAPI surface for the Orpheus streaming-clone service, mirroring
voice_design_dev.py's auth/job-submission shape and intake.py's
"roll back the enqueued state and 503 if spawn fails" pattern.

create_app() takes root/spawn_training/get_engine_cls/store_reload/
store_commit as parameters (rather than importing Modal directly) so this
module is fully unit-testable without Modal -- main.py wires the real Modal
Volume path, .spawn() call, serving engine class, and Volume reload()/commit().

Volume consistency: a Modal Volume is NOT a shared live filesystem -- each
container sees a snapshot, other containers' writes only become visible
after they commit() AND this container reload()s. So every route that reads
voice state calls store_reload() first, and every route that writes voice
state calls store_commit() after (and before any spawn, so the training
container starts from the committed "training" manifest + upload.zip).

Ownership: POST /v1/orpheus-voices accepts an optional "owner" field that is
stored in the manifest. Every per-voice route (including /v1/orpheus-tts)
then requires the same value in the X-Owner header, else 403. Voices with no
stored owner skip the check (backward compatibility only). The caller (the
ReadAloudAI backend) is responsible for setting X-Owner to its
gateway-resolved identity -- this service only enforces the match.
"""
import hmac
import os
import re
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from orpheus_clone_prod.storage import VoiceRecordStore

VOICE_ID_RE = re.compile(r"^v-[0-9a-f]{10}$")
CUSTOM_FAST_PREFIX = "custom-fast:"
REQUIRED_CONSENT_FIELDS = ("speaker_name", "attested_by", "consent", "consent_text_version", "consent_statement")
OWNER_HEADER = "x-owner"
# Only a fresh voice (or one whose previous training failed) may be
# committed. Committing a "ready" voice would re-train into an existing
# merged/ dir and, on failure, mark a working voice "failed"; committing a
# "training" voice would spawn a duplicate job.
COMMITTABLE_STATUSES = ("awaiting_dataset", "failed")


def _noop() -> None:
    return None


def create_app(root: str, spawn_training, get_engine_cls=None, store_reload=None, store_commit=None) -> FastAPI:
    app = FastAPI()
    store = VoiceRecordStore(root=root)
    reload_store = store_reload or _noop
    commit_store = store_commit or _noop

    def _get_engine_cls():
        if get_engine_cls is None:
            raise NotImplementedError("get_engine_cls was not provided to create_app")
        return get_engine_cls()

    def auth(request: Request) -> None:
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        secret = os.environ.get("ORPHEUS_CLONE_SECRET", "")
        if not tok or not secret or not hmac.compare_digest(tok.encode(), secret.encode()):
            raise HTTPException(401, "unauthorized")

    def check_owner(request: Request, status: dict) -> None:
        stored = status.get("owner")
        if stored is None:
            return  # pre-ownership voice: no owner recorded, nothing to enforce
        supplied = request.headers.get(OWNER_HEADER)
        if supplied is None or not hmac.compare_digest(str(supplied).encode(), str(stored).encode()):
            raise HTTPException(403, "forbidden")

    def load_owned_voice(vid: str, request: Request) -> dict:
        """Reload, 404 if unknown/malformed, 403 if owned by someone else."""
        if not VOICE_ID_RE.match(vid):
            raise HTTPException(404, "unknown voice")
        reload_store()
        status = store.read_status(vid)
        if status is None:
            raise HTTPException(404, "unknown voice")
        check_owner(request, status)
        return status

    @app.post("/v1/orpheus-voices")
    async def create_voice(request: Request):
        auth(request)
        body = await request.json()
        missing = [f for f in REQUIRED_CONSENT_FIELDS if not body.get(f)]
        if missing:
            raise HTTPException(400, f"missing required fields: {', '.join(missing)}")
        if body.get("consent") is not True:
            # Must be a real JSON boolean true -- a truthy string like "no"
            # is not consent.
            raise HTTPException(400, "consent must be the boolean true")
        record = {k: body[k] for k in REQUIRED_CONSENT_FIELDS}
        owner = body.get("owner")
        if owner is None:
            # The real caller (ReadAloudAI's backend) never sets "owner" in
            # the body -- it sends its gateway-resolved identity as the
            # X-Owner header on every route, including this one. Fall back
            # to that header so ownership is actually recorded in the
            # deployed flow; explicit body owner (e.g. direct API testing)
            # still takes precedence when present.
            owner = request.headers.get(OWNER_HEADER)
        if owner is not None:
            if not isinstance(owner, str) or not owner:
                raise HTTPException(400, "owner must be a non-empty string")
            record["owner"] = owner
        vid = "v-" + secrets.token_hex(5)
        store.create(vid, record)
        commit_store()
        return {"id": vid, "status": "awaiting_dataset"}

    @app.put("/v1/orpheus-voices/{vid}/dataset")
    async def upload_dataset(vid: str, request: Request):
        auth(request)
        load_owned_voice(vid, request)
        body = await request.body()
        dataset_dir = store.dataset_dir(vid)
        with open(os.path.join(dataset_dir, "upload.zip"), "wb") as f:
            f.write(body)
        store.mark_dataset_uploaded(vid)
        commit_store()
        return {"id": vid, "status": "awaiting_dataset", "uploaded_bytes": len(body)}

    @app.post("/v1/orpheus-voices/{vid}/dataset/commit")
    async def commit_dataset(vid: str, request: Request):
        auth(request)
        status = load_owned_voice(vid, request)
        if status.get("status") not in COMMITTABLE_STATUSES:
            raise HTTPException(
                400, "voice is already ready or training; delete and recreate to retrain"
            )
        if not status.get("dataset_uploaded"):
            raise HTTPException(400, "no dataset uploaded yet")

        store.write_status(vid, "training")
        # Commit BEFORE spawning: the training container mounts the Volume at
        # its latest committed state, so it must see "training" + upload.zip.
        commit_store()
        try:
            spawn_training(vid, root)
        except Exception as e:
            # Roll back to a resumable state, same shape as intake.py's
            # "remove the enqueued marker and 503" pattern -- the customer
            # can retry commit without re-uploading.
            store.write_status(vid, "awaiting_dataset", dataset_uploaded=True)
            commit_store()
            raise HTTPException(503, "training service unavailable, please retry") from e
        return {"id": vid, "status": "training"}

    @app.get("/v1/orpheus-voices/{vid}")
    async def poll_voice(vid: str, request: Request):
        auth(request)
        status = load_owned_voice(vid, request)
        return {"id": vid, **status}

    @app.delete("/v1/orpheus-voices/{vid}")
    async def delete_voice(vid: str, request: Request):
        auth(request)
        status = load_owned_voice(vid, request)
        if status.get("status") == "training":
            # The training container holds its own Volume snapshot and commits
            # it when it finishes; a delete now could be undone by that
            # commit, resurrecting the voice and its consent record.
            raise HTTPException(
                409,
                "cannot delete a voice while it is training; wait for training to finish or fail, then delete",
            )
        store.delete(vid)
        commit_store()
        return {"id": vid, "deleted": True}

    @app.post("/v1/orpheus-tts")
    async def synthesize_voice(request: Request):
        auth(request)
        body = await request.json()
        voice = body.get("voice", "")
        text = body.get("text", "")
        if not text:
            raise HTTPException(400, "text is required")
        if not isinstance(voice, str):
            raise HTTPException(400, "voice must be a string")
        from orpheus_clone_prod.serve import GenerationTimeoutError, UnknownVoiceError, resolve_voice_dir

        # Everything is resolved here in the web container, before paying for
        # a GPU call: ownership (403) first, then resolve_voice_dir -- the
        # single authority on malformed/unknown/not-ready (400). The engine
        # class is parameterized by vid (one voice per GPU container).
        if voice.startswith(CUSTOM_FAST_PREFIX):
            vid = voice[len(CUSTOM_FAST_PREFIX):]
            if VOICE_ID_RE.match(vid):
                reload_store()
                status = store.read_status(vid)
                if status is not None:
                    check_owner(request, status)
        try:
            resolve_voice_dir(voice, store)
        except UnknownVoiceError:
            raise HTTPException(400, f"unknown or not-ready voice: {voice!r}")
        vid = voice[len(CUSTOM_FAST_PREFIX):]

        engine_cls = _get_engine_cls()
        try:
            # .remote() blocks; run it off the event loop.
            chunks = await run_in_threadpool(lambda: engine_cls(vid=vid).synthesize_text.remote(text))
        except UnknownVoiceError:
            raise HTTPException(400, f"unknown or not-ready voice: {voice!r}")
        except GenerationTimeoutError:
            raise HTTPException(503, "generation took too long, please retry")
        return Response(b"".join(chunks), media_type="audio/pcm")

    return app

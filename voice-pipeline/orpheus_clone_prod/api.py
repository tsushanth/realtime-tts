"""
FastAPI surface for the Orpheus streaming-clone service, mirroring
voice_design_dev.py's auth/job-submission shape and intake.py's
"roll back the enqueued state and 503 if spawn fails" pattern.

create_app() takes root/spawn_training as parameters (rather than importing
Modal directly) so this module is fully unit-testable without Modal --
main.py (Task 6) wires the real Modal Volume path and .spawn() call.
"""
import hmac
import os
import re
import secrets

from fastapi import FastAPI, HTTPException, Request

from orpheus_clone_prod.storage import VoiceRecordStore

VOICE_ID_RE = re.compile(r"^v-[0-9a-f]{10}$")
REQUIRED_CONSENT_FIELDS = ("speaker_name", "attested_by", "consent", "consent_text_version", "consent_statement")


def create_app(root: str, spawn_training) -> FastAPI:
    app = FastAPI()
    store = VoiceRecordStore(root=root)

    def auth(request: Request) -> None:
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        secret = os.environ.get("ORPHEUS_CLONE_SECRET", "")
        if not tok or not secret or not hmac.compare_digest(tok.encode(), secret.encode()):
            raise HTTPException(401, "unauthorized")

    @app.post("/v1/orpheus-voices")
    async def create_voice(request: Request):
        auth(request)
        body = await request.json()
        missing = [f for f in REQUIRED_CONSENT_FIELDS if not body.get(f)]
        if missing:
            raise HTTPException(400, f"missing required fields: {', '.join(missing)}")
        vid = "v-" + secrets.token_hex(5)
        store.create(vid, {k: body[k] for k in REQUIRED_CONSENT_FIELDS})
        return {"id": vid, "status": "awaiting_dataset"}

    @app.put("/v1/orpheus-voices/{vid}/dataset")
    async def upload_dataset(vid: str, request: Request):
        auth(request)
        if store.read_status(vid) is None:
            raise HTTPException(404, "unknown voice")
        body = await request.body()
        dataset_dir = store.dataset_dir(vid)
        with open(os.path.join(dataset_dir, "upload.zip"), "wb") as f:
            f.write(body)
        store.mark_dataset_uploaded(vid)
        return {"id": vid, "status": "awaiting_dataset", "uploaded_bytes": len(body)}

    @app.post("/v1/orpheus-voices/{vid}/dataset/commit")
    async def commit_dataset(vid: str, request: Request):
        auth(request)
        status = store.read_status(vid)
        if status is None:
            raise HTTPException(404, "unknown voice")
        if not status.get("dataset_uploaded"):
            raise HTTPException(400, "no dataset uploaded yet")

        store.write_status(vid, "training")
        try:
            spawn_training(vid, root)
        except Exception as e:
            # Roll back to a resumable state, same shape as intake.py's
            # "remove the enqueued marker and 503" pattern -- the customer
            # can retry commit without re-uploading.
            store.write_status(vid, "awaiting_dataset", dataset_uploaded=True)
            raise HTTPException(503, "training service unavailable, please retry") from e
        return {"id": vid, "status": "training"}

    @app.get("/v1/orpheus-voices/{vid}")
    async def poll_voice(vid: str, request: Request):
        auth(request)
        status = store.read_status(vid)
        if status is None:
            raise HTTPException(404, "unknown voice")
        return {"id": vid, **status}

    @app.delete("/v1/orpheus-voices/{vid}")
    async def delete_voice(vid: str, request: Request):
        auth(request)
        if store.read_status(vid) is None:
            raise HTTPException(404, "unknown voice")
        store.delete(vid)
        return {"id": vid, "deleted": True}

    return app

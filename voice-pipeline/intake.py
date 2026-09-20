"""Voice pipeline intake API (Modal web app). Server-to-server: the product backend calls it on
behalf of a signed-in user and is responsible for deciding WHICH user may touch WHICH voice id
(this API only checks one shared secret). Everything is Bearer INTAKE_SECRET.

  POST   /voices                       {owner_user_id, speaker_name, attested_by, consent:true,
                                        consent_text_version?, client_ip?, owner_key_ids?:[...]}
                                        -> {voice_id}. Stores the consent attestation with a timestamp.
                                        owner_user_id (the product user, e.g. a Supabase uuid) is what the Piper
                                        registry uses (owner.json user_ids) so access follows the user across keys.
  GET    /voices?owner_user_id=        list that user's voices [{voice_id, speaker_name, status, created_at, ...}]
  PUT    /voices/{id}/dataset          body = zip (audio .wav/.flac + transcripts: a metadata.csv of
                                        `file|text`, or a <name>.txt next to each audio file). Starts training.
  GET    /voices/{id}                  status: created | training | ready | rejected | deployed (+ manifest/error)
  GET    /voices/{id}/samples/{0-4}    fixed preview sentences (wav)
  POST   /voices/{id}/preview          {text} (<=300 chars) -> wav, synthesized from the trained model
  POST   /voices/{id}/deploy           only when ready: pushes the voice to the Piper serving machine;
                                        returns {voice: "custom:<id>"} for the synthesize `voice` field
  DELETE /voices/{id}                  revoke: removes it from serving and deletes the training data + model

Deploy:  modal deploy train_job.py && modal deploy intake.py   (VOICE_APP_SUFFIX=-x deploys a separate copy)
Secret `voice-intake` must hold INTAKE_SECRET, PIPER_ADMIN_URL, PIPER_ADMIN_TOKEN.
"""
import os as _os
import modal

APP_SUFFIX = _os.environ.get("VOICE_APP_SUFFIX", "")  # must match train_job.py; "" = production names
# baked into the image: the container re-imports this module without the deploy-time environment
image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi==0.109.0", "requests").env({"VOICE_APP_SUFFIX": APP_SUFFIX})
preview_image = modal.Image.debian_slim(python_version="3.11").apt_install("espeak-ng").pip_install("piper-tts==1.8.0", "numpy<2")
app = modal.App("voice-intake" + APP_SUFFIX, image=image)
datasets = modal.Volume.from_name("voice-datasets", create_if_missing=True)
models = modal.Volume.from_name("voice-models", create_if_missing=True)
secret = modal.Secret.from_name("voice-intake")

MAX_ZIP_BYTES = 500 * 1024 * 1024
MAX_FILES = 5000
AUDIO_EXT = (".wav", ".flac", ".ogg", ".opus", ".mp3")  # compressed audio keeps browser uploads small; libsndfile decodes them


@app.function(image=preview_image, volumes={"/models": models}, cpu=2, memory=2048, timeout=120)
def preview_voice(voice_id: str, text: str) -> bytes:
    import io, wave
    from piper import PiperVoice
    models.reload()
    v = PiperVoice.load(f"/models/{voice_id}/model.onnx")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        v.synthesize_wav(text, w)
    return buf.getvalue()


@app.function(secrets=[secret], volumes={"/datasets": datasets, "/models": models}, timeout=900, cpu=1, memory=2048)
@modal.asgi_app()
def api():
    import hmac, io, json, os, re, secrets as pysecrets, shutil, tarfile, time, zipfile

    import requests
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response

    web = FastAPI()
    ID_RE = re.compile(r"^v-[0-9a-f]{10}$")

    def auth(request: Request):
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not tok or not hmac.compare_digest(tok.encode(), os.environ["INTAKE_SECRET"].encode()):
            raise HTTPException(401, "unauthorized")

    def vid_ok(vid: str):
        if not ID_RE.match(vid):
            raise HTTPException(400, "bad voice id")

    def status_of(vid: str) -> dict:
        models.reload()
        m = f"/models/{vid}"
        if os.path.exists(f"{m}/deployed.json"):
            st = "deployed"
        elif os.path.exists(f"{m}/manifest.json"):
            st = "ready"
        elif os.path.exists(f"{m}/error.json"):
            st = "rejected"
        elif os.path.exists(f"{m}/training.json"):
            st = "training"
        elif os.path.exists(f"/datasets/{vid}/consent.json"):
            st = "created"
        else:
            raise HTTPException(404, "unknown voice")
        out = {"voice_id": vid, "status": st}
        try:
            c = json.load(open(f"/datasets/{vid}/consent.json"))
            out["owner_user_id"] = c.get("owner_user_id"); out["speaker_name"] = c.get("speaker_name"); out["created_at"] = c.get("recorded_at")
        except Exception:
            pass
        for name, key in (("manifest.json", "manifest"), ("error.json", "error")):
            if os.path.exists(f"{m}/{name}"):
                out[key] = json.load(open(f"{m}/{name}"))
        if st == "deployed":
            out["voice"] = f"custom:{vid}"
        return out

    @web.post("/voices")
    async def create(request: Request):
        auth(request)
        body = await request.json()
        owners = body.get("owner_key_ids")
        uid = body.get("owner_user_id")
        if owners is not None and not (isinstance(owners, list) and all(isinstance(o, str) for o in owners)):
            raise HTTPException(400, "owner_key_ids must be a list of key ids")
        if not (isinstance(uid, str) and 0 < len(uid) <= 128) and not owners:
            raise HTTPException(400, "owner_user_id (or owner_key_ids) is required")
        if body.get("consent") is not True or not body.get("speaker_name") or not body.get("attested_by"):
            raise HTTPException(400, "consent=true, speaker_name and attested_by are required")
        vid = "v-" + pysecrets.token_hex(5)
        os.makedirs(f"/datasets/{vid}", exist_ok=True)
        json.dump({"consent": True, "speaker_name": body["speaker_name"], "attested_by": body["attested_by"],
                   "owner_user_id": uid, "owner_key_ids": owners or [],
                   "consent_text_version": str(body.get("consent_text_version", ""))[:64],
                   "client_ip": str(body.get("client_ip", ""))[:64], "recorded_at": int(time.time())}, open(f"/datasets/{vid}/consent.json", "w"))
        datasets.commit()
        return {"voice_id": vid}

    @web.get("/voices")
    async def list_voices(request: Request, owner_user_id: str = ""):
        auth(request)
        if not owner_user_id:
            raise HTTPException(400, "owner_user_id is required")
        datasets.reload()
        out = []
        for name in sorted(os.listdir("/datasets")):
            if not ID_RE.match(name):
                continue
            try:
                c = json.load(open(f"/datasets/{name}/consent.json"))
            except Exception:
                continue
            if c.get("owner_user_id") != owner_user_id:
                continue
            try:
                st = status_of(name)
            except HTTPException:
                continue
            m = st.get("manifest") or {}
            out.append({"voice_id": name, "speaker_name": c.get("speaker_name"), "status": st["status"], "created_at": c.get("recorded_at"),
                        "warnings": m.get("warnings", []), "error": st.get("error"), **({"voice": st["voice"]} if "voice" in st else {})})
        return {"voices": out}

    @web.put("/voices/{vid}/dataset")
    async def upload(vid: str, request: Request):
        auth(request); vid_ok(vid)
        if not os.path.exists(f"/datasets/{vid}/consent.json"):
            raise HTTPException(404, "unknown voice")
        if os.path.exists(f"/datasets/{vid}/metadata.csv"):
            raise HTTPException(409, "dataset already uploaded")
        raw, size = io.BytesIO(), 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_ZIP_BYTES:
                raise HTTPException(413, "zip too large")
            raw.write(chunk)
        try:
            z = zipfile.ZipFile(raw)
        except zipfile.BadZipFile:
            raise HTTPException(400, "body must be a zip file")
        infos = [i for i in z.infolist() if not i.is_dir() and not os.path.basename(i.filename).startswith(("._", "."))
                 and "__MACOSX" not in i.filename]
        if len(infos) > MAX_FILES * 2 or sum(i.file_size for i in infos) > 4 * MAX_ZIP_BYTES:
            raise HTTPException(413, "archive too large when extracted")
        wavdir = f"/datasets/{vid}/wavs"
        os.makedirs(wavdir, exist_ok=True)
        texts, audio, csv_rows = {}, [], None
        for i in infos:
            base = os.path.basename(i.filename)
            stem, ext = os.path.splitext(base)
            data = z.read(i)
            if ext.lower() in AUDIO_EXT:
                open(f"{wavdir}/{base}", "wb").write(data); audio.append(base)
            elif ext.lower() == ".txt":
                texts[stem] = data.decode("utf-8", "replace").strip()
            elif base.lower() == "metadata.csv":
                csv_rows = [ln.split("|", 1) for ln in data.decode("utf-8", "replace").splitlines() if "|" in ln]
        rows = csv_rows if csv_rows is not None else [[a, texts[os.path.splitext(a)[0]]] for a in audio if os.path.splitext(a)[0] in texts]
        if not rows:
            shutil.rmtree(wavdir, ignore_errors=True)
            raise HTTPException(400, "no transcripts found: include metadata.csv (file|text) or a .txt beside each audio file")
        with open(f"/datasets/{vid}/metadata.csv", "w") as f:
            for name, text in rows:
                f.write(f"{name}|{text.replace('|', ' ').replace(chr(10), ' ')}\n")
        datasets.commit()
        os.makedirs(f"/models/{vid}", exist_ok=True)
        json.dump({"started_at": int(time.time())}, open(f"/models/{vid}/training.json", "w"))
        models.commit()
        modal.Function.from_name("voice-train" + APP_SUFFIX, "train_voice").spawn(vid)
        return {"voice_id": vid, "status": "training", "clips": len(rows)}

    @web.get("/voices/{vid}")
    async def get(vid: str, request: Request):
        auth(request); vid_ok(vid)
        return status_of(vid)

    @web.get("/voices/{vid}/samples/{n}")
    async def sample(vid: str, n: int, request: Request):
        auth(request); vid_ok(vid)
        if not 0 <= n <= 4:
            raise HTTPException(404, "no such sample")
        models.reload()
        p = f"/models/{vid}/samples/sample_{n}.wav"
        if not os.path.exists(p):
            raise HTTPException(404, "not ready")
        return Response(open(p, "rb").read(), media_type="audio/wav")

    @web.post("/voices/{vid}/preview")
    async def preview(vid: str, request: Request):
        auth(request); vid_ok(vid)
        text = (await request.json()).get("text", "")
        if not text or len(text) > 300:
            raise HTTPException(400, "text must be 1-300 characters")
        if status_of(vid)["status"] not in ("ready", "deployed"):
            raise HTTPException(409, "voice is not trained yet")
        return Response(preview_voice.remote(vid, text), media_type="audio/wav")

    @web.post("/voices/{vid}/deploy")
    async def deploy(vid: str, request: Request):
        auth(request); vid_ok(vid)
        if status_of(vid)["status"] not in ("ready", "deployed"):
            raise HTTPException(409, "voice is not ready")
        consent = json.load(open(f"/datasets/{vid}/consent.json"))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(f"/models/{vid}/model.onnx", arcname="model.onnx")
            tf.add(f"/models/{vid}/model.onnx.json", arcname="model.onnx.json")
            om = {}
            if consent.get("owner_user_id"):
                om["user_ids"] = [consent["owner_user_id"]]
            if consent.get("owner_key_ids"):
                om["key_ids"] = consent["owner_key_ids"]
            owner = json.dumps(om).encode()
            ti = tarfile.TarInfo("owner.json"); ti.size = len(owner)
            tf.addfile(ti, io.BytesIO(owner))
        r = requests.put(f"{os.environ['PIPER_ADMIN_URL']}/admin/voices/{vid}", data=buf.getvalue(),
                         headers={"Authorization": f"Bearer {os.environ['PIPER_ADMIN_TOKEN']}"}, timeout=300)
        if r.status_code != 200:
            raise HTTPException(502, f"serving machine rejected the voice ({r.status_code})")
        json.dump({"deployed_at": int(time.time())}, open(f"/models/{vid}/deployed.json", "w"))
        models.commit()
        return {"voice_id": vid, "voice": f"custom:{vid}"}

    @web.delete("/voices/{vid}")
    async def delete(vid: str, request: Request):
        auth(request); vid_ok(vid)
        r = requests.delete(f"{os.environ['PIPER_ADMIN_URL']}/admin/voices/{vid}",
                            headers={"Authorization": f"Bearer {os.environ['PIPER_ADMIN_TOKEN']}"}, timeout=60)
        if r.status_code != 200:
            raise HTTPException(502, "could not remove the voice from serving; nothing deleted")
        shutil.rmtree(f"/datasets/{vid}", ignore_errors=True)
        shutil.rmtree(f"/models/{vid}", ignore_errors=True)
        datasets.commit(); models.commit()
        return {"deleted": vid}

    return web

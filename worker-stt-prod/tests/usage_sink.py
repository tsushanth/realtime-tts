"""Test-only stand-in for the gateway's /admin/usage/report: records bodies in a modal.Dict so usage reporting
from the worker can be verified end to end. Deploy as `stt-prod-test-sink`, stop afterwards."""
import modal
app = modal.App("stt-prod-test-sink", image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
store = modal.Dict.from_name("stt-prod-test-usage", create_if_missing=True)

@app.function(secrets=[modal.Secret.from_name("stt-prod-test-secret")])
@modal.asgi_app()
def web():
    import os, time
    from fastapi import FastAPI, Request, HTTPException
    api = FastAPI()
    @api.post("/admin/usage/report")
    async def rep(r: Request):
        if r.headers.get("authorization") != f"Bearer {os.environ['MODAL_USAGE_REPORT_SECRET']}":
            raise HTTPException(401)
        store[str(time.time())] = await r.json()
        return {"recorded": True}
    return api

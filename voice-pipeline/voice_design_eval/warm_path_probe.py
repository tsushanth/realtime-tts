import time
import modal

Model = modal.Cls.from_name("voice-design-dev", "VoiceDesignModel")
m = Model()

cases = [
    ("d-warmA1", "calm, warm female voice, slight British accent, professional customer-service tone",
     "Thanks for calling, how can I help you today?"),
    ("d-warmA2", "calm, warm female voice, slight British accent, professional customer-service tone",
     "Let me check that for you, one moment please."),
    ("d-warmA3", "deep, gravelly male voice, slow and deliberate, noir detective narrator",
     "The rain never stopped that whole miserable week."),
]

results = []
for jid, desc, text in cases:
    t0 = time.time()
    r = m.generate.remote(jid, desc, text)
    wall = time.time() - t0
    r["wall_seconds"] = round(wall, 2)
    results.append(r)
    print(r)

print("\n=== SUMMARY ===")
for r in results:
    print(f"{r['job_id']}: wall={r['wall_seconds']}s container_load={r.get('container_load_seconds')}s gen={r['generation_seconds']}s audio={r['audio_seconds']}s")

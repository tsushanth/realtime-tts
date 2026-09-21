"""Modal CPU cold-start probe (temporary app, run with `modal run modal_cold.py`, nothing stays deployed).
Times: spawn() of a ping -> result, for a container that scaled to zero (scaledown_window=2)."""
import modal, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim(python_version="3.11").pip_install("sherpa-onnx==1.13.8", "numpy", "scipy", "fastapi", "uvicorn")
       .add_local_dir(os.path.join(HERE, "..", "docker_models"), "/models"))
app = modal.App("stt-rt-coldprobe")

@app.cls(image=img, cpu=4, memory=2048, scaledown_window=2, timeout=120)
class Stt:
    @modal.enter()
    def load(self):
        t = time.time()
        import sherpa_onnx as so, numpy as np
        d = "/models/sherpa-onnx-streaming-zipformer-en-2023-06-26"; s = "chunk-16-left-128.int8.onnx"
        self.r = so.OnlineRecognizer.from_transducer(tokens=f"{d}/tokens.txt", encoder=f"{d}/encoder-epoch-99-avg-1-{s}",
            decoder=f"{d}/decoder-epoch-99-avg-1-{s}", joiner=f"{d}/joiner-epoch-99-avg-1-{s}", num_threads=1, sample_rate=16000)
        st = self.r.create_stream(); st.accept_waveform(16000, np.zeros(16000, np.float32))
        while self.r.is_ready(st): self.r.decode_stream(st)
        self.load_s = time.time() - t
    @modal.method()
    def ping(self): return {"ready": True, "load_s": self.load_s}

@app.local_entrypoint()
def main():
    for i in range(4):
        time.sleep(20)     # let the previous container scale down
        t = time.time(); r = Stt().ping.spawn().get(); print(f"trial {i}: spawn->ready {time.time()-t:.2f}s (model load in container {r['load_s']:.2f}s)")
        t = time.time(); Stt().ping.remote(); print(f"   warm call {time.time()-t:.3f}s")

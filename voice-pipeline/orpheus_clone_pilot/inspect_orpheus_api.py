"""CPU-only, no-GPU check of the actual OrpheusModel constructor signature
and generate_speech signature -- avoids guessing kwargs one A10G run at a
time after two wrong guesses (tokenizer=, max_model_len=) each burned GPU
minutes in a crash loop before hitting the function timeout.
"""
import modal

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "orpheus-speech", "vllm==0.7.3",
)

app = modal.App("orpheus-api-inspect", image=cpu_image)


@app.function(image=cpu_image, gpu="T4", timeout=120)
def inspect_api():
    import inspect
    from orpheus_tts import OrpheusModel

    print("=== OrpheusModel.__init__ signature ===")
    print(inspect.signature(OrpheusModel.__init__))
    print()
    print("=== OrpheusModel.__init__ source ===")
    try:
        print(inspect.getsource(OrpheusModel.__init__))
    except Exception as e:
        print(f"could not get source: {e}")
    print()
    print("=== OrpheusModel.generate_speech signature ===")
    print(inspect.signature(OrpheusModel.generate_speech))
    print()
    print("=== OrpheusModel.generate_speech source ===")
    try:
        print(inspect.getsource(OrpheusModel.generate_speech))
    except Exception as e:
        print(f"could not get source: {e}")
    print()
    print("=== OrpheusModel._map_model_params source ===")
    try:
        print(inspect.getsource(OrpheusModel._map_model_params))
    except Exception as e:
        print(f"could not get source: {e}")
    print()
    print("=== OrpheusModel module file (for full source dump) ===")
    import importlib
    engine_mod = importlib.import_module(OrpheusModel.__module__)
    print(engine_mod.__file__)
    with open(engine_mod.__file__) as f:
        print(f.read())

    print("=== decoder.py full source ===")
    from orpheus_tts import decoder as decoder_mod
    print(decoder_mod.__file__)
    with open(decoder_mod.__file__) as f:
        print(f.read())
    return "done"


@app.local_entrypoint()
def main():
    inspect_api.remote()

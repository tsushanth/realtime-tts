import struct


def wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM16LE bytes in a WAV container."""
    byte_rate = sample_rate * channels * bits // 8
    header = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
              struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate,
                          channels * bits // 8, bits) +
              b"data" + struct.pack("<I", len(pcm)))
    return header + pcm

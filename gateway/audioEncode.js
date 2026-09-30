// Compressed output for POST /v1/text-to-speech. The worker only speaks PCM/G.711; the gateway
// asks it for pcm_24000 and pipes that through an ffmpeg child process (stdin -> stdout, streaming).
import { spawn } from "node:child_process";

export const FFMPEG_PATH = process.env.FFMPEG_PATH || "ffmpeg";

// format string -> { contentType, sampleRate, args (ffmpeg output options) }
export const COMPRESSED_FORMATS = {
  mp3_24000_64: { contentType: "audio/mpeg", sampleRate: 24000, args: ["-c:a", "libmp3lame", "-b:a", "64k", "-ar", "24000", "-ac", "1", "-f", "mp3"] },
  mp3_24000_128: { contentType: "audio/mpeg", sampleRate: 24000, args: ["-c:a", "libmp3lame", "-b:a", "128k", "-ar", "24000", "-ac", "1", "-f", "mp3"] },
  opus_24000: { contentType: "audio/ogg", sampleRate: 24000, args: ["-c:a", "libopus", "-b:a", "32k", "-application", "voip", "-ar", "24000", "-ac", "1", "-f", "ogg"] },
};

export function isCompressedFormat(fmt) {
  return typeof fmt === "string" && Object.prototype.hasOwnProperty.call(COMPRESSED_FORMATS, fmt);
}

// Spawns ffmpeg reading 24 kHz mono s16le PCM on stdin and writing the encoded stream to stdout.
export function spawnEncoder(fmt) {
  const spec = COMPRESSED_FORMATS[fmt];
  const args = ["-nostdin", "-hide_banner", "-loglevel", "error", "-fflags", "+nobuffer",
    "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
    ...spec.args, "-flush_packets", "1", "pipe:1"];
  return spawn(FFMPEG_PATH, args, { stdio: ["pipe", "pipe", "pipe"] });
}

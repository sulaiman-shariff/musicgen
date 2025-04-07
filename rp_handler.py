import runpod
import time
import io
import base64
import torch
import scipy.io.wavfile
import numpy as np
import librosa

from transformers import MusicgenForConditionalGeneration, AutoProcessor

# ── MODEL & PROCESSOR ──────────────────────────────────────────────────────────

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model.to(device)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

GENERATION_PARAMS = {
    "do_sample": True,
    "guidance_scale": 3,
    "max_new_tokens": 1024,
}

# ── UTILITIES ──────────────────────────────────────────────────────────────────

def tensor_to_wav_base64(audio_tensor, sampling_rate: int) -> str:
    """Convert [1,1,T] tensor to WAV Base64."""
    audio_np = audio_tensor[0, 0].cpu().numpy()
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate=sampling_rate, data=audio_np)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# ── HANDLER ───────────────────────────────────────────────────────────────────

def handler(event):
    input_data = event.get("input", {})
    prompt = input_data.get("prompt", "").strip()
    if not prompt:
        return {"error": "No prompt provided."}

    t0 = time.time()
    try:
        # ─ Audio‑conditioned generation ───────────────────────────────
        if "audio_base64" in input_data:
            # Decode incoming WAV
            audio_bytes = base64.b64decode(input_data["audio_base64"])
            buffer = io.BytesIO(audio_bytes)
            sr_in, audio_data = scipy.io.wavfile.read(buffer)

            # If stereo, take first channel
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]

            # Resample to model’s 32 kHz if needed
            sr_target = model.config.audio_encoder.sampling_rate  # = 32000
            if sr_in != sr_target:
                # Normalize to float32 [-1,1]
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                audio_data = librosa.resample(audio_data, orig_sr=sr_in, target_sr=sr_target)
                sr_in = sr_target
                # Back to int16
                audio_data = (audio_data * 32767).astype(np.int16)

            # Feature‑extract & generate
            proc = processor(
                audio=audio_data,
                sampling_rate=sr_in,
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            padding_mask = proc.get("padding_mask")
            inputs = {k: v.to(device) for k, v in proc.items() if k != "padding_mask"}
            audio_out = model.generate(**inputs, **GENERATION_PARAMS)

            # Decode back to WAV
            if padding_mask is not None:
                decoded = processor.batch_decode(audio_out, padding_mask=padding_mask)[0]
                buf = io.BytesIO()
                scipy.io.wavfile.write(buf, rate=sr_target, data=decoded)
                buf.seek(0)
                out_b64 = base64.b64encode(buf.read()).decode("utf-8")
            else:
                out_b64 = tensor_to_wav_base64(audio_out, sr_target)

        # ─ Text‑only generation ────────────────────────────────────────
        else:
            proc = processor(text=[prompt], padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in proc.items()}
            audio_out = model.generate(**inputs, **GENERATION_PARAMS)
            sr_target = model.config.audio_encoder.sampling_rate
            out_b64 = tensor_to_wav_base64(audio_out, sr_target)

        return {
            "audio_base64": out_b64,
            "generation_time": f"{time.time() - t0:.2f}s"
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

import os
import runpod
import time
import io
import base64
import torch
import scipy.io.wavfile

from transformers import MusicgenForConditionalGeneration, AutoProcessor

# Load the MusicGen model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model.to(device)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# Generation parameters: here we use 1024 new tokens (~20 seconds of audio)
GENERATION_PARAMS = {
    "do_sample": True,
    "guidance_scale": 3,
    "max_new_tokens": 1024,
}

def tensor_to_wav_base64(audio_tensor, sampling_rate: int) -> str:
    """
    Convert the generated audio tensor (assumed shape [batch, channels, seq_length])
    to a base64-encoded WAV file.
    """
    # Use the first sample and its first channel
    audio_np = audio_tensor[0, 0].cpu().numpy()
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate=sampling_rate, data=audio_np)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def handler(event):
    """
    Runpod handler for MusicGen audio generation.

    The input event should be a JSON object containing:
      - "prompt": A text prompt for generation.
      - (Optional) "audio_base64": A base64-encoded WAV file for audio-conditioned generation.

    The output will be a JSON object with:
      - "audio_base64": The generated audio as a base64-encoded WAV file.
      - "generation_time": The time taken to generate the audio.
    """
    input_data = event.get("input", {})
    prompt = input_data.get("prompt", "").strip()
    if not prompt:
        return {"error": "No prompt provided."}

    t0 = time.time()
    try:
        # Audio-conditioned generation if audio_base64 provided
        if input_data.get("audio_base64"):
            audio_bytes = base64.b64decode(input_data["audio_base64"])
            buffer = io.BytesIO(audio_bytes)
            sampling_rate_input, audio_data = scipy.io.wavfile.read(buffer)
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]

            processed_inputs = processor(
                audio=audio_data,
                sampling_rate=sampling_rate_input,
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            padding_mask = processed_inputs.get("padding_mask")
            inputs_device = {k: v.to(device) for k, v in processed_inputs.items() if k != "padding_mask"}
            audio_values = model.generate(**inputs_device, **GENERATION_PARAMS)

            if padding_mask is not None:
                audio_list = processor.batch_decode(audio_values, padding_mask=padding_mask)
                generated_audio = audio_list[0]
                buffer_out = io.BytesIO()
                output_sr = model.config.audio_encoder.sampling_rate
                scipy.io.wavfile.write(buffer_out, rate=output_sr, data=generated_audio)
                buffer_out.seek(0)
                audio_b64_output = base64.b64encode(buffer_out.read()).decode("utf-8")
            else:
                sr = model.config.audio_encoder.sampling_rate
                audio_b64_output = tensor_to_wav_base64(audio_values, sr)
        else:
            # Text-conditional generation
            processed_inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            inputs_device = {k: v.to(device) for k, v in processed_inputs.items()}
            audio_values = model.generate(**inputs_device, **GENERATION_PARAMS)
            sr = model.config.audio_encoder.sampling_rate
            audio_b64_output = tensor_to_wav_base64(audio_values, sr)

        dt = time.time() - t0
        return {"audio_base64": audio_b64_output, "generation_time": f"{dt:.2f}s"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

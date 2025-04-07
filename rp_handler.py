import os
import runpod
import time
import io
import base64
import torch
import scipy.io.wavfile
import tempfile

from transformers import MusicgenForConditionalGeneration, AutoProcessor
from moviepy.editor import VideoFileClip, AudioFileClip  


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
    Runpod handler for MusicGen audio generation and video audio integration.

    The input event should be a JSON object containing:
      - "prompt": A text prompt for generation.
      - (Optional) "audio_base64": A base64-encoded WAV file for audio-conditioned generation.
      - (Optional) "video_base64": A base64-encoded video file (without audio) to which the generated audio will be applied.

    The output will be a JSON object with:
      - If video is provided:
          "video_base64": The final video with audio track (base64-encoded).
      - Otherwise:
          "audio_base64": The generated audio as a base64-encoded WAV file.
      - "generation_time": The time taken to generate the audio (and video processing if applicable).
    """
    input_data = event.get("input", {})
    prompt = input_data.get("prompt", "").strip()
    if not prompt:
        return {"error": "No prompt provided."}

    t0 = time.time()
    try:
        if "audio_base64" in input_data and input_data["audio_base64"]:
            # --- Audio-Prompted Generation ---
            audio_b64_input = input_data["audio_base64"]
            audio_bytes = base64.b64decode(audio_b64_input)
            buffer = io.BytesIO(audio_bytes)
            sampling_rate_input, audio_data = scipy.io.wavfile.read(buffer)
            if audio_data.ndim > 1:
                # If multi-channel, take the first channel
                audio_data = audio_data[:, 0]

            # Process audio and text prompt together
            processed_inputs = processor(
                audio=audio_data,
                sampling_rate=sampling_rate_input,
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            # Save the padding mask (if available) for post-processing
            padding_mask = processed_inputs.get("padding_mask")
            # Move tensor inputs to device (leave non-tensor items on CPU)
            inputs_device = {k: v.to(device) for k, v in processed_inputs.items() if k != "padding_mask"}
            audio_values = model.generate(**inputs_device, **GENERATION_PARAMS)
            
            # Post-process: if a padding mask exists, decode the batch to remove padding.
            if padding_mask is not None:
                audio_list = processor.batch_decode(audio_values, padding_mask=padding_mask)
                generated_audio = audio_list[0]
                buffer_out = io.BytesIO()
                output_sampling_rate = model.config.audio_encoder.sampling_rate
                scipy.io.wavfile.write(buffer_out, rate=output_sampling_rate, data=generated_audio)
                buffer_out.seek(0)
                audio_b64_output = base64.b64encode(buffer_out.read()).decode("utf-8")
            else:
                sampling_rate = model.config.audio_encoder.sampling_rate
                audio_b64_output = tensor_to_wav_base64(audio_values, sampling_rate)
        else:
            # --- Text-Conditional Generation ---
            processed_inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            inputs_device = {k: v.to(device) for k, v in processed_inputs.items()}
            audio_values = model.generate(**inputs_device, **GENERATION_PARAMS)
            sampling_rate = model.config.audio_encoder.sampling_rate
            audio_b64_output = tensor_to_wav_base64(audio_values, sampling_rate)

        # Check if a video is provided to merge the generated audio onto it
        if "video_base64" in input_data and input_data["video_base64"]:
            video_b64 = input_data["video_base64"]
            video_bytes = base64.b64decode(video_b64)

            # Write the video bytes to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_temp_file:
                video_temp_file.write(video_bytes)
                video_temp_filepath = video_temp_file.name

            # Write the generated audio to a temporary WAV file
            audio_bytes = base64.b64decode(audio_b64_output)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp_file:
                audio_temp_file.write(audio_bytes)
                audio_temp_filepath = audio_temp_file.name

            # Load video and audio clips using moviepy
            video_clip = VideoFileClip(video_temp_filepath)
            audio_clip = AudioFileClip(audio_temp_filepath)

            # Align durations by trimming to the shorter clip
            final_duration = min(video_clip.duration, audio_clip.duration)
            video_clip = video_clip.subclip(0, final_duration)
            audio_clip = audio_clip.subclip(0, final_duration)

            # Set the audio for the video clip
            final_clip = video_clip.set_audio(audio_clip)

            # Write the final video to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_video_file:
                output_video_filepath = output_video_file.name
            final_clip.write_videofile(output_video_filepath, codec="libx264", audio_codec="aac", verbose=False, logger=None)

            # Read the final video file and encode it in base64
            with open(output_video_filepath, "rb") as f:
                final_video_bytes = f.read()
            final_video_b64 = base64.b64encode(final_video_bytes).decode("utf-8")

            # Clean up temporary files
            os.remove(video_temp_filepath)
            os.remove(audio_temp_filepath)
            os.remove(output_video_filepath)

            dt = time.time() - t0
            return {"video_base64": final_video_b64, "generation_time": f"{dt:.2f}s"}
        else:
            dt = time.time() - t0
            return {"audio_base64": audio_b64_output, "generation_time": f"{dt:.2f}s"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

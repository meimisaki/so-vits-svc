import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from functools import partial

import gradio as gr
import librosa
import numpy as np
import soundfile

from edgetts.tts_voices import SUPPORTED_LANGUAGES
from inference.infer_tool import Svc

MAXOCTAVE = 2
TEMPDIR = None

def generate_tempfile(suffix=None, prefix=None):
    global TEMPDIR
    _, filepath = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=TEMPDIR)
    return filepath

def find_sovits_model(dirpath):
    for filename in os.listdir(dirpath):
        if filename.endswith(".pth"):
            return os.path.join(dirpath, filename)
    return None

def find_diffusion_model(dirpath):
    for filename in os.listdir(dirpath):
        if filename.startswith("model") and filename.endswith(".pt"):
            return os.path.join(dirpath, filename)
    return None

def find_static_file(dirpath, filename):
    filepath = os.path.join(dirpath, filename)
    return filepath if os.path.exists(filepath) else None

def model_fn(modeldir, model, leakctrl, diffonly, enhancer):
    if model is not None:
        model.unload_model()

    # locate trained models
    sovits_model_path = find_sovits_model(modeldir)
    sovits_config_path = find_static_file(modeldir, "config.json")
    diffusion_model_path = find_diffusion_model(modeldir)
    diffusion_config_path = find_static_file(modeldir, "config.yaml")
    kmeans_model_path = find_static_file(modeldir, "kmeans_10000.pt")
    feature_index_path = find_static_file(modeldir, "feature_and_index.pkl")

    feature_retrieval = leakctrl == "Feature retrieval"
    cluster_model_path = feature_index_path if feature_retrieval else kmeans_model_path

    model = Svc(
        sovits_model_path,
        sovits_config_path,
        cluster_model_path=cluster_model_path,
        feature_retrieval=feature_retrieval,
        diffusion_model_path=diffusion_model_path,
        diffusion_config_path=diffusion_config_path,
        shallow_diffusion=True,
        only_diffusion=diffonly,
        nsf_hifigan_enhance=enhancer,
    )
    speakers = list(model.spk2id.keys())

    return (
        model,
        "Reload Model",
        f"Successfully loaded model into device {str(model.dev)}",
        gr.Dropdown(choices=speakers, value=speakers[0]),
    )

def preset_fn(preset):
    if preset == "Singing":
        f0_predictor = "none"
        leakctrl_ratio = 0.5
    else:
        f0_predictor = "rmvpe"
        leakctrl_ratio = 0
    """
    f0_predictor, pitch_shift, leakctrl_ratio, diff_steps, noise_scale,
    silent_padding, db_threshold, auto_clip, clip_overlap, cross_fade,
    adaptive_key, crepe_f0, loudness_ratio, reencode_audio,
    """
    return (
        f0_predictor, 0, leakctrl_ratio, 100, 0.4,
        0.5, -40, 0, 0, 0.75,
        0, 0.05, 0, False,
    )

def tts_fn(text, gender, lang, rate, volume):
    def to_percent(x):
        return f"+{int(x * 100)}%" if x >= 0 else f"{int(x * 100)}%"

    rate = to_percent(rate)
    volume = to_percent(volume)

    outfile = generate_tempfile(suffix=".wav")
    subprocess.run([sys.executable, "edgetts/tts.py", text, lang, rate, volume, gender, outfile])
    result, orig_sr = librosa.load(outfile)
    os.remove(outfile)

    target_sr = 44100
    resampled = librosa.resample(result, orig_sr=orig_sr, target_sr=target_sr)
    return target_sr, resampled

def inference_fn(
    model, speaker, input_audio,
    f0_predictor, pitch_shift, leakctrl_ratio, diff_steps, noise_scale,
    silent_padding, db_threshold, auto_clip, clip_overlap, cross_fade,
    adaptive_key, crepe_f0, loudness_ratio, reencode_audio,
):
    if model is None:
        return "Error: please load model first", None
    if input_audio is None:
        return "Error: please upload an audio", None

    sample_rate, audio = input_audio
    if np.issubdtype(audio.dtype, np.integer):
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))

    infile = generate_tempfile(suffix=".wav")
    soundfile.write(infile, audio, sample_rate, format="wav")

    result = model.slice_inference(
        infile,
        speaker,
        pitch_shift,
        db_threshold,
        leakctrl_ratio,
        f0_predictor != "none",
        noise_scale,
        pad_seconds=silent_padding,
        clip_seconds=auto_clip,
        lg_num=clip_overlap,
        lgr_num=cross_fade,
        f0_predictor="crepe" if f0_predictor == "none" else f0_predictor,
        enhancer_adaptive_key=adaptive_key,
        cr_threshold=crepe_f0,
        k_step=diff_steps,
        use_spk_mix=False,
        second_encoding=reencode_audio,
        loudness_envelope_adjustment=loudness_ratio,
    )
    model.clear_empty()
    os.remove(infile)

    # gr.Audio force normalize the audio if supplied as a numpy array
    # we must write to a temporary file and return the filepath here
    prefix = f"{speaker}_{f0_predictor}_pitch{pitch_shift}_timbre{leakctrl_ratio}_diff{diff_steps}_"
    outfile = generate_tempfile(suffix=".wav", prefix=prefix)
    soundfile.write(outfile, result, model.target_sample, format="wav")
    return "Success", outfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="so-vits-svc WebUI")
    parser.add_argument("-m", "--model", default="./trained")
    parser.add_argument("-t", "--temp", default="./workspace")
    args = parser.parse_args()

    shutil.rmtree(args.temp, ignore_errors=True)
    os.makedirs(args.temp, exist_ok=True)
    TEMPDIR = args.temp

    with gr.Blocks() as app:

        with gr.Row():
            with gr.Column():
                title = gr.Markdown(value="""${title}""")
            with gr.Column():
                with gr.Accordion(label="About", open=False):
                    about = gr.Markdown(value="""${about}""")

        with gr.Row():
            with gr.Column():
                with gr.Accordion(label="Model setup", open=True):
                    leakctrl = gr.Radio(
                        label="Timbre leakage control method",
                        choices=["Feature retrieval", "K-means clustering"],
                        value="Feature retrieval",
                    )
                    diffonly = gr.Checkbox(label="Diffusion only mode")
                    enhancer = gr.Checkbox(label="NSF-HiFiGAN enhancer (not recommended)")
                    modelptr = gr.State(None)
                    modelbtn = gr.Button(value="Load Model", variant="primary")
                    modelmsg = gr.Textbox(label="Model info")
                    speaker = gr.Dropdown(label="Speaker", interactive=True)

                with gr.Accordion(label="Text to speech", open=False):
                    tts_text = gr.Textbox(label="Text", placeholder="Enter text here")
                    tts_gender = gr.Radio(label="Gender", choices=["Male","Female"], value="Male")
                    tts_lang = gr.Dropdown(label="Language", choices=SUPPORTED_LANGUAGES, value="Auto")
                    tts_rate = gr.Slider(
                        label="Relative speed",
                        minimum=-1, maximum=3, value=0, step=0.1
                    )
                    tts_volume = gr.Slider(
                        label="Relative volume",
                        minimum=-1, maximum=1.5, value=0, step=0.1
                    )
                    tts_btn = gr.Button(value="Synthesize")

                with gr.Accordion(label="Voice conversion", open=True):
                    input_audio = gr.Audio(label="Input audio", type="numpy")
                    inference_btn = gr.Button(value="Inference")
                    output_msg = gr.Textbox(label="Output message")
                    output_audio = gr.Audio(label="Output audio", type="filepath")

            with gr.Column():
                with gr.Accordion(label="Inference options", open=True):
                    inference_preset = gr.Radio(
                        label="Preset",
                        choices=["Singing", "Speaking"],
                        value="Singing",
                        interactive=True,
                    )
                    f0_predictor = gr.Dropdown(
                        label="F0 predictor",
                        choices=["none", "crepe", "dio", "harvest", "pm", "rmvpe"],
                        value="none",
                    )
                    pitch_shift = gr.Slider(
                        label="Pitch shift (in semitones, 12 in an octave)",
                        minimum=-12*MAXOCTAVE, maximum=12*MAXOCTAVE, value=0, step=1,
                    )
                    leakctrl_ratio = gr.Slider(
                        label="Timbre leakage control mix ratio (set to 0 to disable it)",
                        minimum=0, maximum=1, value=0.5, step=0.1,
                    )
                    diff_steps = gr.Slider(
                        label="Shallow diffusion steps",
                        minimum=0, maximum=1000, value=100, step=10,
                    )
                    noise_scale = gr.Slider(
                        label="Noise scale (try NOT to modify this parameter)",
                        minimum=0, maximum=1, value=0.4, step=0.01,
                    )
                    silent_padding = gr.Slider(
                        label="Add silent padding to workaround noise caused by unknown reason (in seconds)",
                        minimum=0, maximum=3, value=0.5, step=0.01,
                    )
                    db_threshold = gr.Slider(
                        label="Silence dB threshold (for slicing audio into chunks)",
                        minimum=-100, maximum=0, value=-40, step=1,
                    )
                    auto_clip = gr.Slider(
                        label="Apply auto clip to reduce memory consumption (in seconds)",
                        minimum=0, maximum=100, value=0, step=1,
                    )
                    clip_overlap = gr.Slider(
                        label="Overlap duration between auto clips (in seconds)",
                        minimum=0, maximum=3, value=0, step=0.01,
                    )
                    cross_fade = gr.Slider(
                        label="Cross fade ratio of overlapping regions",
                        minimum=0, maximum=1, value=0.75, step=0.01,
                    )
                    adaptive_key = gr.Slider(
                        label="Enhancer adaptive key (in semitones, 12 in an octave)",
                        minimum=-12*MAXOCTAVE, maximum=12*MAXOCTAVE, value=0, step=1,
                    )
                    crepe_f0 = gr.Slider(
                        label="CREPE F0 threshold (increase to reduce noise but may result in out-of-tune)",
                        minimum=0, maximum=1, value=0.05, step=0.01,
                    )
                    loudness_ratio = gr.Slider(
                        label="Loudness envelope mix ratio of input and output (0 is input and 1 is output)",
                        minimum=0, maximum=1, value=0, step=0.01,
                    )
                    reencode_audio = gr.Checkbox(
                        label="Re-encode audio before shallow diffusion, with unknown impact on final result"
                    )

        modelbtn.click(
            partial(model_fn, args.model),
            inputs=[modelptr, leakctrl, diffonly, enhancer],
            outputs=[modelptr, modelbtn, modelmsg, speaker],
        )

        inference_preset.change(
            preset_fn,
            inputs=[inference_preset],
            outputs=[
                f0_predictor, pitch_shift, leakctrl_ratio, diff_steps, noise_scale,
                silent_padding, db_threshold, auto_clip, clip_overlap, cross_fade,
                adaptive_key, crepe_f0, loudness_ratio, reencode_audio,
            ],
        )

        tts_btn.click(
            tts_fn,
            inputs=[tts_text, tts_gender, tts_lang, tts_rate, tts_volume],
            outputs=[input_audio],
        )

        inference_btn.click(
            inference_fn,
            inputs=[
                modelptr, speaker, input_audio,
                f0_predictor, pitch_shift, leakctrl_ratio, diff_steps, noise_scale,
                silent_padding, db_threshold, auto_clip, clip_overlap, cross_fade,
                adaptive_key, crepe_f0, loudness_ratio, reencode_audio,
            ],
            outputs=[output_msg, output_audio],
        )

    app.launch(debug=True, share=True)

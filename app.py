import os
import queue
from huggingface_hub import snapshot_download
import hydra

# Download if not exists
os.makedirs("checkpoints", exist_ok=True)
snapshot_download(repo_id="fishaudio/fish-speech-1", local_dir="./checkpoints/fish-speech-1")

print("All checkpoints downloaded")

import html
import os
import threading
from argparse import ArgumentParser
from pathlib import Path

import gradio as gr
import librosa
import torch
from loguru import logger
from transformers import AutoTokenizer

from tools.llama.generate import launch_thread_safe_queue
from tools.vqgan.inference import load_model as load_vqgan_model

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


HEADER_MD = """# Fish Speech

## The demo in this space is version 1.0, Please check [Fish Audio](https://fish.audio) for the best model.
## 该 Demo 为 Fish Speech 1.0 版本, 请在 [Fish Audio](https://fish.audio) 体验最新 DEMO.

A text-to-speech model based on VQ-GAN and Llama developed by [Fish Audio](https://fish.audio).  
由 [Fish Audio](https://fish.audio) 研发的基于 VQ-GAN 和 Llama 的多语种语音合成. 

You can find the source code [here](https://github.com/fishaudio/fish-speech) and models [here](https://huggingface.co/fishaudio/fish-speech-1).  
你可以在 [这里](https://github.com/fishaudio/fish-speech) 找到源代码和 [这里](https://huggingface.co/fishaudio/fish-speech-1) 找到模型.  

Related code are released under BSD-3-Clause License, and weights are released under CC BY-NC-SA 4.0 License.  
相关代码使用 BSD-3-Clause 许可证发布，权重使用 CC BY-NC-SA 4.0 许可证发布.

We are not responsible for any misuse of the model, please consider your local laws and regulations before using it.  
我们不对模型的任何滥用负责，请在使用之前考虑您当地的法律法规.

The model running in this WebUI is Fish Speech V1 Medium SFT 4K.
在此 WebUI 中运行的模型是 Fish Speech V1 Medium SFT 4K.
"""

TEXTBOX_PLACEHOLDER = """Put your text here. 在此处输入文本."""

try:
    import spaces

    GPU_DECORATOR = spaces.GPU
except ImportError:

    def GPU_DECORATOR(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


def build_html_error_message(error):
    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(error)}
    </div>
    """


@GPU_DECORATOR
@torch.inference_mode()
def inference(
    text,
    enable_reference_audio,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    speaker,
):
    if args.max_gradio_length > 0 and len(text) > args.max_gradio_length:
        return None, f"Text is too long, please keep it under {args.max_gradio_length} characters."

    # Parse reference audio aka prompt
    prompt_tokens = None
    if enable_reference_audio and reference_audio is not None:
        # reference_audio_sr, reference_audio_content = reference_audio
        reference_audio_content, _ = librosa.load(
            reference_audio, sr=vqgan_model.sampling_rate, mono=True
        )
        audios = torch.from_numpy(reference_audio_content).to(vqgan_model.device)[
            None, None, :
        ]

        logger.info(
            f"Loaded audio with {audios.shape[2] / vqgan_model.sampling_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=vqgan_model.device, dtype=torch.long
        )
        prompt_tokens = vqgan_model.encode(audios, audio_lengths)[0][0]

    # LLAMA Inference
    request = dict(
        tokenizer=llama_tokenizer,
        device=vqgan_model.device,
        max_new_tokens=max_new_tokens,
        text=text,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=args.compile,
        iterative_prompt=chunk_length > 0,
        chunk_length=chunk_length,
        max_length=args.max_length,
        speaker=speaker if speaker else None,
        prompt_tokens=prompt_tokens if enable_reference_audio else None,
        prompt_text=reference_text if enable_reference_audio else None,
    )

    payload = dict(
        response_queue=queue.Queue(),
        request=request,
    )
    llama_queue.put(payload)

    codes = []
    while True:
        result = payload["response_queue"].get()
        if result == "next":
            # TODO: handle next sentence
            continue

        if result == "done":
            if payload["success"] is False:
                return None, build_html_error_message(payload["response"])
            break

        codes.append(result)

    codes = torch.cat(codes, dim=1)

    # VQGAN Inference
    feature_lengths = torch.tensor([codes.shape[1]], device=vqgan_model.device)
    fake_audios = vqgan_model.decode(
        indices=codes[None], feature_lengths=feature_lengths, return_audios=True
    )[0, 0]

    fake_audios = fake_audios.float().cpu().numpy()

    return (vqgan_model.sampling_rate, fake_audios), None


def build_app():
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'light');window.location.search = params.toString();}}",
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Input Text / 输入文本", placeholder=TEXTBOX_PLACEHOLDER, lines=15
                )

                with gr.Row():
                    with gr.Tab(label="Advanced Config / 高级参数"):
                        chunk_length = gr.Slider(
                            label="Iterative Prompt Length, 0 means off / 迭代提示长度，0 表示关闭",
                            minimum=0,
                            maximum=100,
                            value=30,
                            step=8,
                        )

                        max_new_tokens = gr.Slider(
                            label="Maximum tokens per batch, 0 means no limit / 每批最大令牌数，0 表示无限制",
                            minimum=128,
                            maximum=512,
                            value=512,  # 0 means no limit
                            step=8,
                        )

                        top_p = gr.Slider(
                            label="Top-P", minimum=0, maximum=1, value=0.7, step=0.01
                        )

                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            minimum=0,
                            maximum=2,
                            value=1.5,
                            step=0.01,
                        )

                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0,
                            maximum=2,
                            value=0.7,
                            step=0.01,
                        )

                        speaker = gr.Textbox(
                            label="Speaker / 说话人",
                            placeholder="Type name of the speaker / 输入说话人的名称",
                            lines=1,
                        )

                    with gr.Tab(label="Reference Audio / 参考音频"):
                        gr.Markdown(
                            "5 to 10 seconds of reference audio, useful for specifying speaker. \n5 到 10 秒的参考音频，适用于指定音色。"
                        )

                        enable_reference_audio = gr.Checkbox(
                            label="Enable Reference Audio / 启用参考音频",
                        )
                        reference_audio = gr.Audio(
                            label="Reference Audio / 参考音频",
                            type="filepath",
                        )
                        reference_text = gr.Textbox(
                            label="Reference Text / 参考文本",
                            placeholder="参考文本",
                            lines=1,
                        )

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(label="Error Message / 错误信息")
                with gr.Row():
                    audio = gr.Audio(label="Generated Audio / 音频", type="numpy")

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001F3A7 Generate / 合成", variant="primary"
                        )

        # # Submit
        generate.click(
            inference,
            [
                text,
                enable_reference_audio,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                speaker,
            ],
            [audio, error],
            concurrency_limit=1,
        )

    return app


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/text2semantic-sft-large-v1-4k.pth",
    )
    parser.add_argument(
        "--llama-config-name", type=str, default="dual_ar_2_codebook_large"
    )
    parser.add_argument(
        "--vqgan-checkpoint-path",
        type=Path,
        default="checkpoints/vq-gan-group-fsq-2x1024.pth",
    )
    parser.add_argument("--vqgan-config-name", type=str, default="vqgan_pretrain")
    parser.add_argument("--tokenizer", type=str, default="fishaudio/fish-speech-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.precision = torch.half if args.half else torch.bfloat16
    args.compile = True
    args.max_gradio_length = 1024
    args.tokenizer = "./checkpoints/fish-speech-1"
    args.llama_checkpoint_path = "./checkpoints/fish-speech-1/text2semantic-sft-medium-v1-4k.pth"
    args.llama_config_name = "dual_ar_2_codebook_medium"
    args.vqgan_checkpoint_path = "./checkpoints/fish-speech-1/vq-gan-group-fsq-2x1024.pth"
    args.vqgan_config_name = "vqgan_pretrain"

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        config_name=args.llama_config_name,
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        max_length=args.max_length,
        compile=args.compile,
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info("Llama model loaded, loading VQ-GAN model...")

    vqgan_model = load_vqgan_model(
        config_name=args.vqgan_config_name,
        checkpoint_path=args.vqgan_checkpoint_path,
        device=args.device,
    )

    logger.info("VQ-GAN model loaded, warming up...")

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    inference(
        text="Hello, world!",
        enable_reference_audio=False,
        reference_audio=None,
        reference_text="",
        max_new_tokens=0,
        chunk_length=0,
        top_p=0.7,
        repetition_penalty=1.5,
        temperature=0.7,
        speaker=None,
    )

    logger.info("Warming up done, launching the web UI...")

    app = build_app()
    app.launch(show_api=False)

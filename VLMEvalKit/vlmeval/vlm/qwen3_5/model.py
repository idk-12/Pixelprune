from __future__ import annotations
import os
import threading
import time

import torch

from vlmeval.smp import get_gpu_memory
from ..base import BaseModel
from ..qwen3_vl.prompt import Qwen3VLPromptMixin


def _loading_timer(label: str):
    stop = threading.Event()
    start = time.time()

    def _run():
        while not stop.wait(timeout=3):
            print(f'\r  {label} ... {time.time() - start:.0f}s', end='', flush=True)
        print(f'\r  {label} done ({time.time() - start:.1f}s)' + ' ' * 10, flush=True)

    threading.Thread(target=_run, daemon=True).start()
    return stop.set


class Qwen3_5Chat(Qwen3VLPromptMixin, BaseModel):
    """VLMEvalKit wrapper for Qwen3_5ForConditionalGeneration (hybrid-attention VLM).

    Uses processor.apply_chat_template(tokenize=True, return_dict=True) which
    produces mm_token_type_ids required by M-RoPE without any qwen_vl_utils dependency.
    """

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        use_custom_prompt: bool = True,
        enable_thinking: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.enable_thinking = enable_thinking
        self.verbose = verbose
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems else -1
        assert max_gpu_mem > 0, 'No GPU memory found'

        name = os.path.basename(model_path.rstrip('/'))
        _stop = _loading_timer(f'Loading {name}')
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map='auto',
            attn_implementation='flash_attention_2',
        )
        _stop()
        self.model.eval()
        torch.cuda.empty_cache()

    def _prepare_messages(self, message, dataset=None):
        content = []
        for s in message:
            if s['type'] == 'image':
                # Use raw file path — load_image() handles os.path.isfile() paths directly.
                # Do NOT use ensure_image_url (file:// prefix breaks PIL loading).
                item = {'type': 'image', 'image': s['value']}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f'Unsupported message type: {s["type"]}')
            content.append(item)
        return [{'role': 'user', 'content': content}]

    def generate_inner(self, message, dataset=None):
        from PIL import Image
        messages = self._prepare_messages(message, dataset=dataset)
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        # Step 1: 只生成文本模板，不触碰图片文件
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        # Step 2: 用 PIL 加载图片，对格式有问题的 PNG/zlib 容忍度更高
        images = [Image.open(s['value']).convert('RGB') for s in message if s['type'] == 'image']

        proc_kw = {}
        if self.min_pixels is not None:
            proc_kw['min_pixels'] = self.min_pixels
        if self.max_pixels is not None:
            proc_kw['max_pixels'] = self.max_pixels

        inputs = self.processor(
            text=[text],
            images=images or None,
            return_tensors='pt',
            **proc_kw,
        )
        inputs = inputs.to(self.model.device)

        _IMAGE_TOKEN_ID = 151655 if not hasattr(self.model.config, 'image_token_id') \
            else self.model.config.image_token_id
        _n_img = int((inputs.input_ids == _IMAGE_TOKEN_ID).sum())
        _n_txt = inputs.input_ids.shape[1] - _n_img
        print(f'[Tokens] image={_n_img}  text={_n_txt}  total={inputs.input_ids.shape[1]}')

        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response


class Qwen3_5ChatMMTok(Qwen3_5Chat):
    """Qwen3_5Chat with MMTok training-free visual token compression.

    Patches Qwen3_5Model.forward via mmtok_qwen3_5_vl() after loading.
    retain_ratio controls the fraction of visual tokens kept (e.g. 0.5 = 50%).
    """

    def __init__(self, model_path: str, retain_ratio: float = 0.5, **kwargs):
        super().__init__(model_path, **kwargs)

        from mmtok.qwen.qwen3_5_mmtok import mmtok_qwen3_5_vl
        mmtok_qwen3_5_vl(
            self.model,
            self.processor.tokenizer,
            self.processor,
            retain_ratio=retain_ratio,
        )
        self.retain_ratio = retain_ratio

# all_model_agent_3_do_sample_false.py 에 이어서
import copy
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForVision2Seq
from llava.model.builder import load_pretrained_model
import torch
from decord import cpu, VideoReader
from torch import distributed as dist
from tqdm import tqdm
import json
import torch.nn.functional as F

from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from all_model_util import *
import torch
import os
import json
import random
import argparse
import time
import pandas as pd
import re
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from qwen_vl_utils import process_vision_info

import sys
sys.path.append("/data1/lgagent_0402/model_ckpt/InternVL3-8B")

import conversation as internvl_conversation


def get_anno(anno_path):
    # return sample_idx, anno [0]
    anno = json.load(open(anno_path, 'r'))
    return anno[0]

class InternVL8B:
    def __init__(self, device_id=0, path='/data1/lgagent_0402/model_ckpt/InternVL3-8B'):
        self.device_id = device_id
        self.device    = f"cuda:{device_id}"
        self.modelpath = path
        # device_map = self.split_model('InternVL3-8B')
        self.model = AutoModel.from_pretrained(
            self.modelpath,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelpath, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=64, do_sample=False)
    
    def get_model_name(self):
        return 'intern_8b'

    def build_transform(self,input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=336, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
            'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        device_map['language_model.model.rotary_emb'] = 0

        return device_map
    
    def load_video(self, video_path, frame_indices, input_size=448, max_num=1):

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        # torch.Size([48, 3, 448, 448])
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def chat(self, model, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, return_logits=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id

        template = internvl_conversation.get_conv_template(model.template)
        template.system_message = model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(model.device)
        attention_mask = model_inputs['attention_mask'].to(model.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=return_logits,
            output_scores=return_logits,
            **generation_config
        )
        
        if return_logits: # logits is return
            response = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)[0]
            choice_tokens = ["A", "B", "C", "D", "E"]
            choice_token_ids = [
                                self.tokenizer.encode(c, add_special_tokens=False)[0]
                                for c in choice_tokens
                            ]
            first_step_logits = generation_output.scores[0][0]
            choice_logits = {
                            c: first_step_logits[token_id].item()
                            for c, token_id in zip(choice_tokens, choice_token_ids)
                        }
            choice_probs = F.softmax(first_step_logits[choice_token_ids], dim=-1)
            choice_probs_dict = {
                c: p.item() for c, p in zip(choice_tokens, choice_probs)
            }
            scores_cpu = [s.cpu().to(torch.float16) for s in generation_output.scores]
        else:
            response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))

        if return_logits:
            return response, history, scores_cpu, choice_probs_dict

        if return_history:
            return response, history, None, None
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    def get_answer(self, video_path, query, sorted_frame_idx, return_logits=False):

        pv, np_list = self.load_video(video_path, frame_indices=sorted_frame_idx)
        pv = pv.to(torch.bfloat16).to(self.device)
        question = "".join(
            [f"Frame{i+1}: <image>\n" for i in range(len(np_list))]
        ) + query
        # gen_cfg = dict(self.generation_config, do_sample=do_sample)
        gen_cfg = dict(self.generation_config)
        response, _, logits, choice_probs = self.chat(
            self.model, self.tokenizer, pv, question, gen_cfg,
            num_patches_list=np_list, history=None, return_history=True, return_logits=return_logits
        )
        del pv
        torch.cuda.empty_cache()
        
        if not return_logits:
            return response
        return {
            "text": response,
            "option_probs": choice_probs,
            "logits": logits,
        }
    
    def get_text_answer(self, query):

        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, None, query, self.generation_config,
                                    num_patches_list=None, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response


class Eagle25Agent:
    def __init__(self, device_id=1):
        # TODO: change path
        model_path = "/data1/lgagent_0402/model_ckpt/Eagle2.5-8B/"
        self.device_id = device_id
        self.device    = f"cuda:{device_id}"
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self.device,   # ← 단일 GPU 고정 (inter-GPU 통신 제거)
        ).eval()
        used = torch.cuda.memory_allocated(device_id) / 1e9

        # ── generation_config 패치 ──────────────────────────────
        # language_model의 use_cache=False가 기본값으로 설정되어 있어
        # 매 토큰마다 전체 시퀀스 재계산 → 느림 + 빈 출력 원인
        # use_cache=True로 강제 설정
        if hasattr(self.model, 'language_model'):
            from transformers import GenerationConfig
            self.model.language_model.generation_config = GenerationConfig(
                bos_token_id=151643,
                eos_token_id=151645,
                pad_token_id=151645,
                use_cache=True,       # ← KV cache 활성화
            )
            self.model.generation_config = GenerationConfig(
                bos_token_id=151643,
                eos_token_id=151645,
                pad_token_id=151645,
                use_cache=True,
            )
        print(f"[Eagle2.5] Ready on {self.device}. VRAM: {used:.1f}GB")

    def get_model_name(self):
        return 'eagle25_8b'

    def _build_inputs(self, messages: list) -> dict:
        """
        Eagle2.5 공식 인터페이스:
          1. apply_chat_template → text 문자열
          2. process_vision_info → image/video 텐서
          3. processor(text=..., images=..., videos=...) 로 합치기
             ↑ 이때 processor는 tokenizer가 아닌 multimodal processor임
               images/videos 인자를 직접 받음 (tokenizer와 다름)
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # process_vision_info: Eagle2.5 processor 내장 메서드
        try:
            image_inputs, video_inputs, video_kwargs = \
                self.processor.process_vision_info(messages, return_video_kwargs=True)
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
                padding=True,
                **({} if video_kwargs is None else {"videos_kwargs": video_kwargs}),
            )
        except TypeError:
            # return_video_kwargs 미지원 구버전 fallback
            image_inputs, video_inputs = self.processor.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
                padding=True,
            )
        return inputs

    def get_answer(self, video_path: str, prompt: str,
                   sample_idx, do_sample: bool = False, return_logits=False) -> str:
        
        messages = [{
            "role": "user",
            "content": [
                {
                "type": "video", 
                "video": video_path,
                "nframes": len(sample_idx)
                },   
                {"type": "text",  "text": prompt},
            ],
        }]
        
        text_list = [self.processor.apply_chat_template( messages, tokenize=False, add_generation_prompt=True )]
        image_inputs, video_inputs, video_kwargs = self.processor.process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True, videos_kwargs=video_kwargs).to(self.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                                                **inputs, 
                                                max_new_tokens=128, 
                                                do_sample=False,
                                                return_dict_in_generate=return_logits,
                                                output_scores=return_logits,
                                                )
        if return_logits:
            scores = generated_ids.scores
            scores_cpu = [s.cpu().to(torch.float16) for s in scores]
            choice_tokens = ["A", "B", "C", "D", "E"]
            choice_token_ids = [
                                self.processor.tokenizer.encode(c, add_special_tokens=False)[0]
                                for c in choice_tokens
                            ]
            first_step_logits = scores[0][0]
            choice_logits = {
                            c: first_step_logits[token_id].item()
                            for c, token_id in zip(choice_tokens, choice_token_ids)
                        }
            choice_probs = F.softmax(first_step_logits[choice_token_ids], dim=-1)
            choice_probs_dict = {
                c: p.item() for c, p in zip(choice_tokens, choice_probs)
            }
            generated_ids = generated_ids.sequences

        output_text = self.processor.batch_decode( generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False )
        raw_text = output_text[0] if output_text else ""
        del inputs, generated_ids
        torch.cuda.empty_cache()

        if not return_logits:
            return raw_text
        return {
            "text": raw_text,
            "option_probs": choice_probs_dict,
            "logits": scores_cpu,
        }
    
    def get_text_answer(self, prompt: str) -> str:
        # Eagle2.5: model.generate()는 멀티모달 입력 전용
        # 텍스트 전용은 language_model을 직접 사용해야 함
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor.tokenizer(
            text, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            # generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids = self.model.language_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                use_cache=True,
                do_sample=False,
            )
        # inputs_embeds 기반 생성: output에 입력 포함 안됨 → trimming 불필요
        output_text = self.processor.batch_decode( generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False )
        del inputs, generated_ids
        torch.cuda.empty_cache()
        return output_text[0] if output_text else ""


class Qwen3_8bAgent:
    # TODO: change path
    def __init__(self):
        model_path = '/data1/lgagent_0402/model_ckpt/Qwen3-VL-8B-Instruct'
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, 
                                                            torch_dtype=torch.bfloat16, 
                                                            attn_implementation="flash_attention_2", 
                                                            device_map="auto")

    def get_model_name(self):
        return 'qwen3vl_8b'

    def get_text_answer(self, text):
        messages = [[{
            "role": "user", 
            "content": [{"type": "text", "text": text}]
            }]]
        _first_dev = next(self.model.parameters()).device
        images, videos = None, None
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(_first_dev)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=64,  do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def get_answer(self, video_path, text, sample_idx=None, multi_image_path= None, return_logits=False):

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
        native_fps = vr.get_avg_fps()
        step = sample_idx[1] - sample_idx[0] if len(sample_idx) > 1 else 1
        effective_sample_fps = native_fps / step
        video = [Image.fromarray(vr[i].asnumpy()) for i in sample_idx]
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", 
                 "video": video, 
                 "sample_fps": effective_sample_fps},
                {"type": "text",  "text": text},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], 
                                        return_video_kwargs=True, 
                                        image_patch_size= 16,
                                        return_video_metadata=True)
        _first_dev = next(self.model.parameters()).device
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt").to(_first_dev)
        with torch.inference_mode():
            output_ids = self.model.generate(
                                            **inputs, 
                                            max_new_tokens=64,  
                                            do_sample=False,
                                            return_dict_in_generate=return_logits,
                                            output_scores=return_logits
                                            )
            torch.cuda.empty_cache()
        
        if return_logits:
            scores = output_ids.scores
            scores_cpu = [s.cpu().to(torch.float16) for s in scores]
            choice_tokens = ["A", "B", "C", "D", "E"]
            choice_token_ids = [
                                self.processor.tokenizer.encode(c, add_special_tokens=False)[0]
                                for c in choice_tokens
                            ]
            first_step_logits = scores[0][0]
            choice_logits = {
                            c: first_step_logits[token_id].item()
                            for c, token_id in zip(choice_tokens, choice_token_ids)
                        }
            choice_probs = F.softmax(first_step_logits[choice_token_ids], dim=-1)
            choice_probs_dict = {
                c: p.item() for c, p in zip(choice_tokens, choice_probs)
            }
            output_ids = output_ids.sequences

        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if not return_logits:
            return output_text
        return {
            "text": output_text,
            "option_probs": choice_probs_dict,
            "logits": scores_cpu,
        }

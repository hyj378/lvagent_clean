
import os
import random
import json
from all_model_agent_3 import  InternVL8B, Eagle25Agent, Qwen3_8bAgent
from all_model_util import *
from tqdm import tqdm
import copy
import time
import pandas as pd
import re
from decord import VideoReader, cpu
import threading 
from PIL import Image
import torchvision.transforms as T
import re
import argparse

# 加载clip模型
import torch
import torch.nn as nn
import numpy as np
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP4Clip

SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                 "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 가능한 범위에서 재현성 강화
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch가 허용하면 deterministic 강제
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def make_agent_rng(base_seed: int, anno_idx: int, agent_name: str, round_tag: str):
    seed_str = f"{base_seed}-{anno_idx}-{agent_name}-{round_tag}"
    seed = abs(hash(seed_str)) % (2**32)
    return random.Random(seed)

def init_model(model_path, args):
    model_state_dict = torch.load(model_path, map_location='cpu')
    model = CLIP4Clip.from_pretrained("cross-base", cache_dir="", state_dict=model_state_dict, task_config=args)
    model.to('cuda')
    return model

def _get_text(tokenizer, video_id, sentence):
    choice_video_ids = [video_id]
    n_caption = len(choice_video_ids)
    k = n_caption
    pairs_text = np.zeros((k, 77), dtype=np.int64)
    pairs_mask = np.zeros((k, 77), dtype=np.int64)
    pairs_segment = np.zeros((k, 77), dtype=np.int64)

    words = tokenizer.tokenize(sentence)

    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = 76
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < 77:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    pairs_text[0] = np.array(input_ids)
    pairs_mask[0] = np.array(input_mask)
    pairs_segment[0] = np.array(segment_ids)
    pairs_text = torch.Tensor(pairs_text).cuda()
    pairs_mask = torch.Tensor(pairs_mask).cuda()
    pairs_segment = torch.Tensor(pairs_segment).cuda()
    return pairs_text.long(), pairs_mask.long(), pairs_segment.long(), choice_video_ids

def extract_and_resize_frames(video_path, frame_indices):
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(frame_indices).asnumpy()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    resized_frames = []
    for frame in frames:
        resized_frame = transform(frame)
        resized_frames.append(resized_frame)
    resized_frames = torch.stack(resized_frames)
    return resized_frames

def run_clip4clip(model, video_path, text, sample_idx):
    # 定义 args 字典
    args = {
        'video_dim': 1024,
        'max_words': 60,
        'max_frames': 16,
        'feature_framerate': 1,
        'margin': 0.1,
        'hard_negative_rate': 0.5,
        'negative_weighting': 1,
        'n_pair': 1,
        'text_num_hidden_layers': 16,
        'visual_num_hidden_layers': 16,
        'cross_num_hidden_layers': 4,
        'linear_patch': "2d",
        'sim_header': "seqTransf"
    }
    tokenizer = ClipTokenizer()
    input_ids, input_mask, segment_ids, choice_video_ids = _get_text(tokenizer, video_path, text)
    video = extract_and_resize_frames(video_path, sample_idx).cuda()
    video_mask = torch.Tensor([[1] * 16]).cuda()
    token_type_ids = torch.Tensor([[0] * 16]).cuda()
    visual_output = model.get_visual_output(video, video_mask=video_mask, shaped=True, video_frame=16)
    text_feat = model.get_sequence_output(input_ids, segment_ids, input_mask, shaped=True)
    b1b2_logits, *_tmp = model.get_similarity_logits(text_feat, visual_output, input_mask, video_mask,
                                                     loose_type=True, eval='myeval')
    return b1b2_logits

model_path = './pytorch_model_0.0011.bin.25'
BASE_SEED=42
args = {
    'video_dim': 1024,
    'max_words': 60,
    'max_frames': 16,
    'feature_framerate': 1,
    'margin': 0.1,
    'hard_negative_rate': 0.5,
    'negative_weighting': 1,
    'n_pair': 1,
    'text_num_hidden_layers': 16,
    'visual_num_hidden_layers': 16,
    'cross_num_hidden_layers': 4,
    'linear_patch': "2d",
    'sim_header': "seqTransf"
}
asp_clip = init_model(model_path, args)



def get_max_frame_block(video_path, text_prompt, sample_frame=16, sample_dict = None, rng=None):
    best_clip_score_idx = []
    best_clip_score = 0
    select_block = 0
    sample_all_frames = []


    for i in range(1, 7):
        if sample_dict:
            sorted_frame_idx = sample_dict['all_samp'][i - 1]
        else:
            sorted_frame_idx = get_frame_idx_path(video_path, i, sample_frame, rng=rng)
        clip_score = run_clip4clip(asp_clip, video_path, text_prompt, sorted_frame_idx)
        if clip_score > best_clip_score:
            best_clip_score = clip_score
            best_clip_score_idx = sorted_frame_idx
            select_block = i
        sample_all_frames.append(sorted_frame_idx)

    return best_clip_score_idx, select_block, sample_all_frames

def get_frame_idx_path(video_path, round = 0, sample_frame=16, judge_whole=False, rng=None):
    # 计算每一帧和text的相似度，然后选出最相似的帧，返回这些帧的路径
    rng = rng or random
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    if len(vr) <= 96 or judge_whole:
        sorted_frame_idx = rng.sample(range(len(vr)), sample_frame)
        sorted_frame_idx = sorted(sorted_frame_idx)
        return sorted_frame_idx
    inter = len(vr) // 6
    if 1<= round <= 6:
        sp = round * inter
        try:
            sorted_frame_idx = rng.sample(range(sp-inter, sp - 1), sample_frame)
        except:
            sorted_frame_idx = rng.sample(range(len(vr)), sample_frame)
    else:
        sorted_frame_idx = rng.sample(range(len(vr)), sample_frame)

    sorted_frame_idx = sorted(sorted_frame_idx)
    return sorted_frame_idx

def get_result_first_round(agent_set, anno, anno_idx, base_seed = 42, return_logits=False):
    answer_dict = {}
    sample_dict = {}
    logits_dict = {}
    option_logits_dict = {}

    decide_watch, info_prompt, get_mme_answer = get_lvbench_prompt(anno)
    video_path = os.path.join('videos/longvideobench/videos', anno['video_path'])

    def process_agent(agent):
        agent_name = agent.get_model_name()
        agent_rng = make_agent_rng(base_seed, anno_idx, agent_name, "first_round")
        watch = agent.get_answer(video_path, decide_watch, anno[agent_name]["watch_samp"])
        watch = watch[0] if isinstance(watch, list) else watch
        anno[agent_name]['watch'] = watch
        if 'Yes' in watch:
            if 'sample_idx' in anno[agent_name].keys():
                sample_idx = anno[agent_name]['sample_idx']
            else:
                sample_idx = get_frame_idx_path(video_path, round=0, sample_frame=16, rng=agent_rng)
            result = agent.get_answer(video_path, get_mme_answer, sample_idx, return_logits=return_logits)
            text_prompt = agent.get_answer(video_path, info_prompt, anno[agent_name]["watch_samp"])
            anno[agent_name]['info'] = text_prompt
        else:
            # text_prompt_ori = anno[agent_name]['info']
            text_prompt = agent.get_answer(video_path, info_prompt, anno[agent_name]["watch_samp"])
            anno[agent_name]['info'] = text_prompt

            if isinstance(text_prompt, list): text_prompt = text_prompt[0]
            if 'sample_dict' in anno[agent_name].keys():
                best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, 16, anno[agent_name]['sample_dict'], rng=agent_rng)
            else:
                best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, sample_frame=16, rng=agent_rng)

            result = agent.get_answer(video_path, get_mme_answer, sample_all_frames[select_block - 1], return_logits=return_logits)
        
        if isinstance(result, dict):
            result_text = result["text"]
            logits = result["logits"]
            option_logits = result["option_probs"]
            logits_dict[agent_name] = logits
            option_logits_dict[agent_name] = option_logits
        else:
            result_text = result

        if isinstance(result_text, list):
            result_text = result_text[0]

        answer_dict[agent_name] = result_text.split('Answer: ')[-1][0]
        if 'Yes' not in watch and answer_dict[agent_name] == chr(ord('A') + anno['correct_choice']):
            sample_dict[agent_name] = {'all_samp': sample_all_frames, 'block': select_block}

    threads = []
    for agent in agent_set:
        # process_agent(agent)
        thread = threading.Thread(target=process_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    
    answer_set = set(answer_dict.values())
    print(answer_dict)

    if not return_logits:
        return answer_set, answer_dict, sample_dict, None, None
    else:
        return answer_set, answer_dict, sample_dict, logits_dict, option_logits_dict

def get_result_second_round(agent_set, anno, history_info=None, anno_idx=0, base_seed=42, return_logits=False):
    answer_dict = {}
    sample_dict = {}
    logits_dict = {}
    option_logits_dict = {}
    decide_watch, info_prompt, get_mme_answer = get_lvbench_prompt(anno)

    video_path = os.path.join('videos/longvideobench/videos', anno['video_path'])

    def process_agent(agent):
        agent_name = agent.get_model_name()
        agent_rng = make_agent_rng(base_seed, anno_idx, agent_name, "second_round")
        watch = anno[agent_name]['watch'][0] if isinstance(anno[agent_name]['watch'], list) else anno[agent_name]['watch']

        if 'Yes' in watch:

            sample_idx = get_frame_idx_path(video_path, round=0, sample_frame=16, rng=agent_rng)
            result = agent.get_answer(video_path, get_mme_answer, sample_idx, return_logits=return_logits)
        else:
            text_prompt = history_info[agent_name]
            if isinstance(text_prompt, list): text_prompt = text_prompt[0]
            if agent_name == 'intern_78b':
                if 'sample_dict' in anno[agent_name].keys():
                    best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, 16, anno[agent_name]['sample_dict'], rng=agent_rng)
                else:
                    best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, sample_frame=16, rng=agent_rng)

                result = agent.get_answer(video_path, get_mme_answer, sample_all_frames[select_block - 1], return_logits=return_logits)
            else:
                best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, sample_frame=16, rng=agent_rng)

            result = agent.get_answer(video_path, get_mme_answer, sample_all_frames[select_block - 1], return_logits=return_logits)

        if result is None:
            result_text = 'N'

        if isinstance(result, list):
            result_text = result[0] if result else 'N'

        if not isinstance(result, str):
            result_text = str(result)

        if isinstance(result, dict):
            result_text = result["text"]
            logits = result["logits"]
            option_logits = result["option_probs"]
            logits_dict[agent_name] = logits
            option_logits_dict[agent_name] = option_logits

        result_text = result_text.strip()

        match = re.search(r'Answer\s*:\s*([A-E])\b', result_text, re.IGNORECASE)
        if match:
            pred = match.group(1).upper()
        else:
            match = re.search(r'\b([A-E])\b', result_text.upper())
            pred = match.group(1) if match else 'N'
        answer_dict[agent_name] = pred

        if 'Yes' not in watch and answer_dict[agent_name] == chr(ord('A') + anno['correct_choice']):
            sample_dict[agent_name] = {'all_samp': sample_all_frames, 'block': select_block}
  

    threads = []
    for agent in agent_set:
        thread = threading.Thread(target=process_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    answer_set = set(answer_dict.values())
    if not return_logits:
        return answer_set, answer_dict, sample_dict, None, None
    else:
        return answer_set, answer_dict, sample_dict, logits_dict, option_logits_dict

def reason_process(agent_set, anno, answer_dict, sample_idx=None, history_info=None, anno_idx=0, base_seed=42):
    video_path = os.path.join('videos/longvideobench/videos', anno['video_path'])
    
    ans_dict = {}

    def process_agent(agent):
        agent_name = agent.get_model_name()
        agent_rng = make_agent_rng(base_seed, anno_idx, agent_name, "reason_process")

        reason_prompt = "Given the video frames you've seen, and the question along with your answer, deeply analyze the logical steps and evidence from the frames that led you to provide this particular answer. The Question is: {}\n, The predict answer is {}\n.".format(
            anno['question'], anno['candidates'][ord(answer_dict[agent_name]) - ord('A')])
        if 'sample_idx' not in anno[agent_name]:
            rand_block = agent_rng.randint(1, 6)
            local_sample_idx = get_frame_idx_path(video_path, round=rand_block, sample_frame=16, judge_whole=True, rng=agent_rng)
        else:
            local_sample_idx = anno[agent_name]['sample_idx']
        try:
            result = agent.get_answer(video_path, reason_prompt, local_sample_idx)
        except:
            print(anno['video_path'])
            rand_block = agent_rng.randint(1, 6)
            local_sample_idx = get_frame_idx_path(video_path, round=rand_block, sample_frame=16, judge_whole=True, rng=agent_rng)
            result = agent.get_answer(video_path, reason_prompt, local_sample_idx)
        if isinstance(result, list):
            result = result[0]
        ans_dict[agent_name] = result

    threads = []
    for agent in agent_set:
        thread = threading.Thread(target=process_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return ans_dict

def parse_json(text):
    if isinstance(text, list):
        text = text[0]
    text = re.sub(r"[\n\t]", "", text)
    text = text.replace('```json', '').replace('```', '')
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        print("No valid JSON found in the text.")
        return None

def agent_back_process(agent_set, data):
    scores = {}

    for model in data.keys():
        score = 0
        for sub_dict in data.values():
            if model in sub_dict:
                score += int(sub_dict[model])
        scores[model] = score
    
    # priority_order = ['intern_8b', 'llava_72b', 'intern_78b']
    priority_order = ['intern_8b', 'qwen3vl_8b', 'eagle25_8b']
    min_score = min(scores.values())
    lowest_score_keys = [key for key, score in scores.items() if score == min_score]
    if len(lowest_score_keys) > 1:
        for priority_key in priority_order:
            if priority_key in lowest_score_keys:
                lowest_score_key = priority_key
                break
    else:
        lowest_score_key = lowest_score_keys[0]

    # 3. 排除掉分数最低的字典
    new_data = [key for key in data.keys() if key != lowest_score_key]

    return new_data, lowest_score_key, scores

def discuss_text_process(agent_set, anno, answer_dict, reason_dict):
    all_agent_name = [agent.get_model_name() for agent in agent_set]
    other_agent_name = copy.deepcopy(all_agent_name)
    sys_prompt = ""
    for key in reason_dict.keys():
        if isinstance(reason_dict[key], list):
            reason_dict[key] = reason_dict[key][0]
    discuss_dict = {}
    def process_agent(agent):
        agent_name = agent.get_model_name()
        local_other_agent_name = copy.deepcopy(other_agent_name)
        local_other_agent_name.remove(agent_name)
        if len(local_other_agent_name) == 1:
            answer_format = {agent_name: "1-10", local_other_agent_name[0]: "1-10"}
            discuss_prompt = f"""Given the answers and the reasoning for judgment from this model and two other models, please rate this model and the other two models. The score ranges from 1-10. Output in dictionary format.
            The question is: {anno['question']}, 
            The answer of this model is {answer_dict[agent_name]}, the reason is {reason_dict[agent_name]}.
            The answer of {local_other_agent_name[0]} model is {answer_dict[local_other_agent_name[0]]}, the reason is {reason_dict[local_other_agent_name[0]]}.
            You do not need to explain your answer, just give me scores as your answer following the answer_format.
            Please strictly follow the answer format! The answer_format is:
            {answer_format}
            """
        if len(local_other_agent_name) > 1:
            answer_format = {agent_name: "1-10", local_other_agent_name[0]: "1-10", local_other_agent_name[1]: "1-10"}
            discuss_prompt = f"""Given the answers and the reasoning for judgment from this model and two other models. 
            The question is: {anno['question']}
            The answer of this model is {answer_dict[agent_name]}, the reason is {reason_dict[agent_name]}.
            The answer of {local_other_agent_name[0]} model is {answer_dict[local_other_agent_name[0]]}, the reason is {reason_dict[local_other_agent_name[0]]}.
            The answer of {local_other_agent_name[1]} model is {answer_dict[local_other_agent_name[1]]}, the reason is {reason_dict[local_other_agent_name[1]]}.
            Please score the performance of this model an other two models base on their reasoning. The score ranges from 1-10. Output in dict format.
            You do not need to explain your answer, just give me scores as your answer following the answer_format.
            Please strictly follow the answer format! The answer_format is:
            {answer_format}
            """

        if agent_name == 'llava_72b':
            video_path = os.path.join('videos/longvideobench/videos', anno['video_path'])
            temp = agent.get_answer(video_path, discuss_prompt, anno['llava_72b']['watch_samp'])
        else:
            temp = agent.get_text_answer(discuss_prompt)
        print(f"results is {temp} || type: {type(temp)}")
        temp = parse_json(temp)
        is_valid = True
        
        try:
            for key, value in temp.items():
                temp[key] = int(value)
        except:
            tmp = copy.deepcopy(temp)
            temp = {}
            for lo_agent_name_i in local_other_agent_name:
                temp[lo_agent_name_i] = 'no_val'  
            temp[agent_name] = 'no_val'
            

        for value in temp.values():
            if not isinstance(value, int) or value < 1 or value > 10:
                is_valid = False
                break
        if is_valid:
            discuss_dict[agent_name] = temp
        else:
            discuss_dict[agent_name] = {'eagle25_8b': 8, 'intern_8b': 6, 'qwen3vl_8b': 7}

    threads = []
    for agent in agent_set:
        process_agent(agent)
        # thread = threading.Thread(target=process_agent, args=(agent,))
        # threads.append(thread)
        # thread.start()

    for thread in threads:
        thread.join()

    return discuss_dict

def generate_history_info(agent_set, anno, new_data, lowest_score_key, scores, reason_dict, answer_dict):

    all_prompt = " Discussion History Summary:\n"
    for data in new_data:
        all_prompt += "{}'s answer: {}\n Reason: {}\n The final score is {}.\n".format(data, anno['candidates'][ord(answer_dict[data]) - ord('A')], reason_dict[data], scores[data])
    all_prompt += "Removed Answer ({})\n Answer: {}\n Reason {}\n However, this reason was deemed unconvincing, so this answer was removed from the discussion.".format(
        lowest_score_key, anno['candidates'][ord(answer_dict[lowest_score_key]) - ord('A')], reason_dict[lowest_score_key])
    # 请从这些history中提取出要回答这个问题需要什么关键信息。
    # 给之前的info, 问题，答案
    
    history_info = {}
    for agent in agent_set:
        agent_name = agent.get_model_name()
        
        history_generate_prompt = lvbench_info_history(anno, all_prompt, anno[agent_name]['info'])

        if agent_name == 'llava_72b':
            video_path = os.path.join('videos/longvideobench/videos', anno['video_path'])
            history_info[agent_name] = agent.get_answer(video_path, history_generate_prompt, anno['llava_72b']['watch_samp'])
        else:
            history_info[agent_name] = agent.get_text_answer(history_generate_prompt)
        
        if isinstance(history_info[agent_name], list): history_info[agent_name] = history_info[agent_name][0] 
    return history_info

def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./lvbench_anno_stable_0406.json")
    parser.add_argument("--save_logits_path", type=str, default="./logits")
    parser.add_argument("--anno_path", type=str, default="./anno_org/lvbench_anno.json")
    parser.add_argument("--return_logits", action='store_true')
    
    outargs    = parser.parse_args()
    if not os.path.exists(outargs.save_logits_path):
        os.makedirs(outargs.save_logits_path)
        os.makedirs(os.path.join(outargs.save_logits_path, "first"))
        os.makedirs(os.path.join(outargs.save_logits_path, "second"))
        os.makedirs(os.path.join(outargs.save_logits_path, "third"))

    internvl8b = InternVL8B()
    eagle7b = Eagle25Agent()
    qwen3vl8b = Qwen3_8bAgent()

    all_num = 0
    correct = 0
    all_agent_set = [internvl8b, eagle7b, qwen3vl8b]
    anno_data = json.load(open(outargs.anno_path))
    for idx, anno in tqdm(enumerate(anno_data), desc="processing items"):
        agent_set = all_agent_set
        all_num += 1
        print(anno['video_path'])
        print(anno['question'])
        print(anno['candidates'])
        
        for agent_i in agent_set:
            agent_name_i = agent_i.get_model_name()
            if agent_name_i not in anno.keys():
                anno[agent_name_i] = {}
            if 'watch_samp' not in anno[agent_name_i].keys():
                anno[agent_name_i]['watch_samp'] = None
            if 'watch' not in anno[agent_name_i].keys():
                anno[agent_name_i]['watch'] = None
            if 'info' not in anno[agent_name_i].keys():
                anno[agent_name_i]['info'] = None
        
        # (First round)
        answer_set, answer_dict, sample_dict, logits_dict, option_logits_dict = get_result_first_round(agent_set, anno, anno_idx=idx, base_seed=BASE_SEED, return_logits=outargs.return_logits)
        anno['first_samp'] = sample_dict
        print(f"\nThe answer is : {chr(ord('A') + anno['correct_choice'])}")
        
        anno['first_round'] = {}
        if outargs.return_logits:
            for k_, v_ in logits_dict.items():
                pt_save_path = os.path.join(outargs.save_logits_path, "first", f"{idx}_{k_}.pt")
                torch.save(logits_dict[k_][0], pt_save_path)
            anno['first_round']['logits_path'] = pt_save_path
            anno['first_round']['option_logits_dict'] = option_logits_dict

        # majority cutting
        if len(answer_set) < 3:
            values = list(answer_dict.values())
            selected_answer = None
            for ans in answer_set:
                if values.count(ans) >= 2:
                    selected_answer = ans
                    break
            # save both final answer and the agents that produced it
            final_agents = [agent_name for agent_name, pred in answer_dict.items() if pred == selected_answer]
            anno['final_round'] = 1
            anno['final_agent'] = final_agents
            anno['final_answer'] = str(selected_answer)
            if anno['final_answer'] == chr(ord('A') + anno['correct_choice']):
                correct += 1
            print("\n[End at 1st round] correct/total: {}/{}, Acc: {}\n".format(correct, idx+1, correct / (idx+1)))
            write_json(anno_data, outargs.save_path)

        reason_dict = reason_process(agent_set, anno, answer_dict, anno_idx=idx, base_seed=BASE_SEED)
        discuss_dict = discuss_text_process(agent_set, anno, answer_dict, reason_dict)
        new_data, lowest_score_key, scores = agent_back_process(agent_set, discuss_dict)
        history_info = generate_history_info(agent_set, anno, new_data, lowest_score_key, scores, reason_dict, answer_dict)
        anno['first_round']['answer_dict'] = answer_dict
        anno['first_round']['scores'] = scores
        anno['first_round']['reason_dict'] = reason_dict
        anno['first_round']['discuss_dict'] = discuss_dict
        anno['first_round']['history_info'] = history_info


        new_agent_set = []
        for agent in agent_set:
            if agent.get_model_name() in new_data:
                new_agent_set.append(agent)
        agent_set = new_agent_set
        
        # (Second round)
        answer_set, answer_dict, sample_dict, logits_dict, option_logits_dict = get_result_second_round(agent_set, anno, history_info, anno_idx=idx, base_seed=BASE_SEED, return_logits=outargs.return_logits)
        anno['second_samp'] = sample_dict
        print(f"\nThe answer is : {chr(ord('A') + anno['correct_choice'])}")
        print(answer_set)

        anno['second_round'] = {}
        if outargs.return_logits:
            for k_, v_ in logits_dict.items():
                pt_save_path = os.path.join(outargs.save_logits_path, "second", f"{idx}_{k_}.pt")
                torch.save(logits_dict[k_][0], pt_save_path)
            anno['second_round']['logits_path'] = pt_save_path
            anno['second_round']['option_logits_dict'] = option_logits_dict

        if len(answer_set) == 1:
            answer_set = next(iter(answer_set))
            answer_set = str(answer_set)
            if answer_set == chr(ord('A') + anno['correct_choice']):
                correct += 1
            # save both final answer and the agents that produced it
            final_agents = [agent_name for agent_name, pred in answer_dict.items()]
            anno['final_round'] = 2
            anno['final_agent'] = final_agents
            anno['final_answer'] = str(answer_set)
            print("\n[End at 2nd round] correct/total: {}/{}, Acc: {}\n".format(correct, idx+1, correct / (idx+1)))
            write_json(anno_data, outargs.save_path)
            continue
        
        print(answer_dict, chr(ord('A') + anno['correct_choice']))
        reason_dict = reason_process(agent_set, anno, answer_dict, anno_idx=idx, base_seed=BASE_SEED)
        discuss_dict = discuss_text_process(agent_set, anno, answer_dict, reason_dict)
        new_data, lowest_score_key, scores = agent_back_process(agent_set, discuss_dict)
        history_info = generate_history_info(agent_set, anno, new_data, lowest_score_key, scores, reason_dict, answer_dict)
        anno['second_round']['answer_dict'] = answer_dict
        anno['second_round']['scores'] = scores
        anno['second_round']['reason_dict'] = reason_dict
        anno['second_round']['discuss_dict'] = discuss_dict
        anno['second_round']['history_info'] = history_info
        new_agent_set = []
        for agent in agent_set:
            if agent.get_model_name() in new_data:
                new_agent_set.append(agent)
        agent_set = new_agent_set

        # (third round)
        answer_set, answer_dict, sample_dict, logits_dict, option_logits_dict = get_result_second_round(agent_set, anno, history_info, return_logits=outargs.return_logits)
        anno['third_samp'] = sample_dict
        anno['third_round'] = {}  
        if outargs.return_logits:
            for k_, v_ in logits_dict.items():
                pt_save_path = os.path.join(outargs.save_logits_path, "third", f"{idx}_{k_}.pt")
                torch.save(logits_dict[k_][0], pt_save_path)
            anno['third_round']['logits_path'] = pt_save_path
            anno['third_round']['option_logits_dict'] = option_logits_dict
        anno['third_round']['answer_dict'] = answer_dict
        final_answer = answer_dict[agent_set[0].get_model_name()]        


        anno['final_round'] = 3
        final_agents = [agent_name for agent_name, pred in answer_dict.items()]
        anno['final_agent'] = final_agents
        anno['final_answer'] = str(final_answer)
        if final_answer == chr(ord('A') + anno['correct_choice']):
            correct += 1
        
        print(answer_dict, chr(ord('A') + anno['correct_choice']))
        print("\n[End at 3th round] correct/total: {}/{}, Acc: {}\n".format(correct, idx+1, correct / (idx+1)))
        write_json(anno_data, outargs.save_path)
    
    write_json(anno_data, outargs.save_path)
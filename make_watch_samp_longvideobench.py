import json
import os
from decord import VideoReader, cpu
import random
from tqdm import tqdm
def generate_watch_samp(video_path, num_frames=4):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    idx = random.sample(range(total_frames), min(num_frames, total_frames))
    return sorted(idx)

def read_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def write_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f)

def main():
    json_path = '/data1/lgagent_0402/anno_org/lvbench_anno.json'
    save_path = '/data1/lgagent_0402/anno_org/lvbench_anno.json'
    missing_path = '/data1/lgagent_0402/missing_videos.json'
    missing = []
    num_missing = 0
    data = read_json(json_path)
    for idx, item in tqdm(enumerate(data)):
        video_path = os.path.join('/data1/LongVideoBench/videos', item['video_path'])
        if os.path.exists(video_path):
            for agent_name in ['intern_8b', 'qwen3vl_8b', 'eagle25_8b']:
                data[idx][agent_name] = {}
                data[idx][agent_name]['watch_samp'] = generate_watch_samp(video_path)
        else:
            print(f"Video path {video_path} does not exist")
            missing.append(video_path)
            num_missing += 1
            
    
    write_json(save_path, data)
    write_json(missing_path, missing)
    print(f"===== Missing video numbers: {num_missing}")

if __name__ == '__main__':
    main()
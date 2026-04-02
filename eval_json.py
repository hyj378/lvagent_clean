import os
import json
from glob import glob

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    results_path = glob('/data1/LVAgent_git/outputs0330_16frame/*.json')
    for path in results_path:
        name = path.split('/')[-1][:-5]
        data = read_json(path)
        correct = 0
        for data_i in data:        
            try:
                if chr(ord('A') + data_i['correct_choice']) == data_i['final_answer']:
                    correct += 1
            except:
                pass
        print(f"[{name}] Acc: {correct/len(data)*100:.2f} ({correct}/{len(data)})")
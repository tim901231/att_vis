import json
import random

fname = "../../dataset/cub_caption/annotations/test.json"
out = "./test.json"

with open(fname) as f:
    data = json.load(f)
print(data[0])

sampled_size = 100
data = random.sample(data, sampled_size)
print(data[0])

with open(out, 'w') as f:
    json.dump(data, f, indent=4)
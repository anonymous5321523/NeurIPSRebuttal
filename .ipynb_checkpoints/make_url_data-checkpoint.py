import pickle
import glob
import numpy as np
import random
import requests
from io import BytesIO
from PIL import Image

with open('/home/jovyan/fileviewer/ChunsanHong/repetition/ConZIC/data_refinement/sd1.4_v1_testset.pickle','rb') as f: 
    prompts = pickle.load(f)
with open('/home/jovyan/fileviewer/ChunsanHong/repetition/ConZIC/data_refinement/curated_data/sd_rep_new/urls.pickle','rb') as f: 
    url_2 = pickle.load(f)
with open('/home/jovyan/fileviewer/ChunsanHong/repetition/ConZIC/data_refinement/curated_data/my_data/urls.pickle','rb') as f: 
    url_1 = pickle.load(f)

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

fail_ids = []

with open('alter_urls_2.pkl','rb') as f: alter_url = pickle.load(f)
#for key in alter_url: url_2[key] = alter_url[key]

#for i, url in enumerate(url_2):
    #try:
    #    Image.open(BytesIO(requests.get(url,headers=headers, timeout = 5, verify = False).content)).convert('RGB')
    #except:
        #if i not in [8, 12, 40, 46,53,54, 55, 56, 57, 58, 59,60, 73, 74, 83, 89, 128]: fail_ids.append(i)
        #fail_ids.append(i)

#breakpoint()

dir_folders = '/home/jovyan/fileviewer/ChunsanHong/repetition/neurips_rebuttal/exps_mitigation/base/*'
folders = sorted(glob.glob(dir_folders))[:-1]

src = [f for f in glob.glob('/home/jovyan/fileviewer/ChunsanHong/repetition/ConZIC/data_refinement/curated_data/*/*') if not f.endswith('pickle')]

exclude_id = [f'sd_rep_new/{i}' for i in [8, 12, 40, 46,53,54, 55, 56, 57, 58, 59,60, 73, 74, 83, 89, 128]]
exclude_id += [f'my_data/{i}' for i in [32, 33, 37]]

for prompt, folder in zip(prompts.keys(), folders):
    with open(f'{folder}/metadata.pickle','rb') as f: res = pickle.load(f)
    candidates = list(set(np.where((res['SSCD']>0.35))[1].tolist()))
    candidates = ['/'.join(src[c].split('/')[-2:]).split('.')[0] for c in candidates]
    candidates += prompts[prompt]
    candidates = [c for c in candidates if (c not in exclude_id) and (not c.startswith('sd2'))]
    tmp_urls = []
    for candidate in candidates:
        img_num = int(candidate.split('/')[1].split('.')[0])
        if candidate.startswith('my_data'): tmp_urls.append(url_1[img_num])
        if candidate.startswith('sd_rep_new'): tmp_urls.append(url_2[img_num])
    tmp_urls = set(tmp_urls)
    prompts[prompt] = tmp_urls

reverse_dict ={}
for prompt in prompts:
    urls = prompts[prompt]
    for url in urls:
        if url not in reverse_dict: 
            reverse_dict[url] = [prompt]
        else:
            reverse_dict[url].append(prompt)

cnt = [len(reverse_dict[url]) for url in reverse_dict]
tmp_urls = list(reverse_dict.keys())

drop_urls = [tmp_urls[i] for i in np.where(np.array(cnt)>150)[0]]
drop_prompts = []
for url in drop_urls:
    drop_prompts += reverse_dict[url]
drop_prompts = list(set(drop_prompts))
random.seed(42)
drop_prompts = random.sample(drop_prompts, 264)
for prompt in drop_prompts: del prompts[prompt]

prompts = list(prompts.items())
random.shuffle(prompts)
prompts = dict(prompts)
#with open('prompts_url.pickle','wb') as f: pickle.dump(prompts, f)
breakpoint()
print(1)
import os
import pickle
import glob
import requests
import torch
import warnings
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from serpapi import GoogleSearch
import serpapi
from concurrent.futures import ThreadPoolExecutor, as_completed
import open_clip
from aesthetic.model import aesthetic_predictor
from optim_utils import get_SSCD_feature, measure_CLIP_similarity
from io_utils import *
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
import ast

# 경고 메시지 억제
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

def download_image(url, timeout=3):
    try:
        response = requests.get(url, verify=False, timeout=timeout)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        return None

def download_images_in_parallel(image_urls, max_workers=4, timeout=3):
    images = [None] * len(image_urls)  # 결과를 저장할 리스트 초기화

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # futures 리스트 생성
        futures = {executor.submit(download_image, url, timeout): idx for idx, url in enumerate(image_urls)}

        # 결과를 순서대로 받아오기
        for future in as_completed(futures):
            idx = futures[future]
            try:
                images[idx] = future.result()
            except Exception as e:
                images[idx] = None

    return images

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

session = requests.Session()
retries = Retry(
    total=5,  # 최대 재시도 횟수
    backoff_factor=1,  # 재시도 간 대기 시간 (지수 증가)
    status_forcelist=[500, 502, 503, 504],  # 재시도할 상태 코드
    raise_on_status=False  # 상태 코드에 따라 에러 발생하지 않음
)
adapter = HTTPAdapter(max_retries=retries)
session.mount('https://', adapter)

model_type = 'SD1'

device = 'cuda'

dir_sim_model = 'sscd_disc_large.torchscript.pt'
sim_model = torch.jit.load(dir_sim_model).to('cuda')

reference_model = "ViT-g-14"
reference_model_pretrain = "laion2b_s12b_b42k"
ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
    reference_model,
    pretrained=reference_model_pretrain,
    device=device,
)
ref_tokenizer = open_clip.get_tokenizer(reference_model)

if model_type == 'SD1':
    dataset = pd.read_csv('SD1_final.csv')
    dataset['urls'] = dataset['urls'].apply(ast.literal_eval)
    exp_name = 'exp_final'
    os.makedirs(f'{exp_name}/google', exist_ok=True)
elif model_type == 'SD2':
    breakpoint()
# prompts = [p for p in prompts.keys()]

total_max_sim = []
total_clip_score = []
first_max_sim = []
first_clip_score = []

max_sims = []
max_clip_score = []

SSCD = 0
CLIP = 0

progress_bar = tqdm(range(len(dataset)), desc="Processing", unit="prompt")

for i in progress_bar:
    if os.path.exists(f'{exp_name}/google/img_{str(i).zfill(6)}/metadata.pickle'):
        with open(f'{exp_name}/google/img_{str(i).zfill(6)}/metadata.pickle', 'rb') as f: metadata = pickle.load(f)
        sim = metadata['SSCD']
        clip_score = metadata['CLIP']
        max_sims.append(sim[clip_score.argmax()].max())
        max_clip_score.append(clip_score[clip_score.argmax()])
        continue

    prompt = dataset.iloc[i]['prompt']
    gt_urls = dataset.iloc[i]['urls']

    params = {
      "q": prompt,
      "engine": "google_images",
      "ijn": "0",
      "api_key": "8ae4d395b2177aaa29c0b0a9c1ad95e5ff1a1ff2de99e805429bdbad4ac5800e"
    }
    search = GoogleSearch(params)
    # search = serpapi.search(params_go_here)
    results = search.get_dict()
    images_results = results.get("images_results", [])
    
    image_urls = [res['original'] for res in images_results if 'original' in res.keys()]
    gen_images = download_images_in_parallel(image_urls, max_workers=64, timeout=1)

    image_urls = [image_urls[idx] for idx, img in enumerate(gen_images) if img is not None]
    gen_images = [img for img in gen_images if img is not None]

    if len(gen_images) == 0: continue
    gen_feats = get_SSCD_feature(gen_images, sim_model, device)


    gt_images = []
    for url in gt_urls:
        try:
            response = session.get(url, verify=False, headers=headers)
            response.raise_for_status()  # 상태 코드가 200이 아니면 예외 발생
            image = Image.open(BytesIO(response.content)).convert('RGB')
            gt_images.append(image)
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve image from {url}: {e}")
            breakpoint()
    gt_feats = get_SSCD_feature(gt_images, sim_model, device)


    sim = torch.mm(gen_feats, gt_feats.T).cpu().numpy()

    
    clip_score = measure_CLIP_similarity(gen_images, prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device).cpu().numpy()

    first_max_sim.append(sim[0].max())
    first_clip_score.append(clip_score[0])

    max_sims.append(sim[clip_score.argmax()].max())
    max_clip_score.append(clip_score[clip_score.argmax()])

    metadata = {'urls': image_urls, 'SSCD': sim, 'CLIP': clip_score, 'prompt': prompt}

    os.makedirs(f'{exp_name}/google/img_{str(i).zfill(6)}', exist_ok=True)
    with open(f'{exp_name}/google/img_{str(i).zfill(6)}/metadata.pickle', 'wb') as f: 
        pickle.dump(metadata, f)

    SSCD = sum(max_sims) / len(max_sims)
    CLIP = sum(max_clip_score) / len(max_clip_score)

    tqdm.write(f"Current SSCD: {SSCD}, Current CLIP: {CLIP}")

print(f"Final SSCD: {SSCD}")
print(f"Final CLIP: {CLIP}")

with open(f'{exp_name}/google/img_{str(i).zfill(6)}/result.pickle', 'wb') as f: 
    pickle.dump({'SSCD': SSCD, 'CLIP': CLIP}, f)
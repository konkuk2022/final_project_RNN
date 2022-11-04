import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import argparse
import gc

from utils import cos_similiarity, kss_sentence, one_hot_encoder
from model import ELECTRALSTMClassification
from transformers import AutoTokenizer


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = argparse.ArgumentParser()

    args.add_argument("--embedding_size", type=int, default=768)
    args.add_argument("--num_labels", type=int, default=44)
    args.add_argument("--dropout", type=float, default=0.1)
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--max_length", type=int, default=512)
    args.add_argument("--n_epochs", type=int, default=20)
    args.add_argument("--lr", type=float, default=1e-5)
    args.add_argument("--device", type=str, default=device)
    args.add_argument("--threshold", type=float, default=0.3)
    args.add_argument("--early_stopping_patience", type=int, default=10)
    args.add_argument("--resume_from")

    config = args.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained('beomi/KcELECTRA-base', do_lower_case=False)
    
    # 모델 pth 위치
    MODEl_PATH = './model/best_model.pth' 
    
    model = ELECTRALSTMClassification(config)
    model.load_state_dict(torch.load(MODEl_PATH)['model_state_dict'],strict=False)
    model.to(device)
    
    diary = {}
    
    
    # 문장 입력 -> SEP 토큰대입 -> token화
    diary['text'] = input('일기 : ')
    # diary['sep_text'] = kss_sentence(diary['text'])
    diary_embedding = tokenizer(kss_sentence(diary['text']), 
                         truncation=True, 
                         max_length=config.max_length, 
                         padding="max_length", 
                         return_token_type_ids=False, 
                         return_attention_mask=True, 
                         add_special_tokens=True)
    
    # 모델에 대입 후 감정 추출
    
    input_id = torch.LongTensor(diary_embedding['input_ids']).unsqueeze(0).to('cuda')
    mask = torch.LongTensor(diary_embedding['attention_mask']).unsqueeze(0).to('cuda')
    sep_idx = torch.where(input_id == 3)
    
    diary_pred,_,_ = model(input_id, mask, sep_idx)
    
    # 확률 값으로 변환하기 위한 sigmoid
    diary['pb_emotion'] = torch.sigmoid(diary_pred[0])
    diary['emotion'] = one_hot_encoder(diary['pb_emotion'])
    
    LABELS = ['불평/불만', '환영/호의', '감동/감탄', '지긋지긋', '고마움',
              '슬픔', '화남/분노', '존경', '기대감', '우쭐댐/무시함',
              '안타까움/실망', '비장함', '의심/불신', '뿌듯함', '편안/쾌적',
              '신기함/관심', '아껴주는', '부끄러움', '공포/무서움', '절망',
              '한심함', '역겨움/징그러움', '짜증', '어이없음', '없음',
              '패배/자기혐오', '귀찮음', '힘듦/지침', '즐거움/신남', '깨달음',
              '죄책감', '증오/혐오', '흐뭇함(귀여움/예쁨)', '당황/난처', '경악',
              '부담/안_내킴', '서러움', '재미없음', '불쌍함/연민', '놀람',
              '행복', '불안/걱정', '기쁨', '안심/신뢰']
    
    # movie 데이터 
    # [제목, 줄거리, 0.4이상 감정, 감정 확률벡터]
    # []
    movie_path = './data/movie/영화_52_data.pkl'

    movie_data = pd.read_pickle(movie_path)
    
    
    # 영화와 감정 cosine 유사도 계산
    movie_data['cos_sim'] = movie_data['pb_emotion'].map(lambda x: cos_similiarity(np.array(x), np.array(diary['pb_emotion'].detach().cpu())))
    
    movie_data.sort_value(by='cos_sim', ascending=False)
    
    # 결과 출력(일기, 감정, 영화 추천, 영화 감정)
    print(diary['text'])
    for l, p in zip(LABELS, diary['pb_emotion'].tolist()):
        if p>= 0.4:
            print(f"{l}, {p}")
    
    print(movie_data.head)

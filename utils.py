import kss
import numpy as np

# cosine similarity
def cos_similiarity(v1, v2):
    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1)))*np.sqrt(sum(np.square(v2))))
    similarity = dot_product/l2_norm

    return similarity



# 여러 문장을 나눠서 sep 토큰을 대입
def kss_sentence(sent):
    x = ''
    split_sent = kss.split_sentences(sent)
    for i,s in enumerate(split_sent):
        if i == 0:
            x = s
        else:
            x += ' [SEP] ' + s
    return x

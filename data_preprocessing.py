import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
# print('전체 샘플의 개수 :', len(lines))

lines = lines.loc[:, 'src':'tar']
lines = lines[0:60000]  # 6만개만 저장
# print(lines.sample(10))

lines.tar = lines.tar.apply(lambda x: '\t ' + x + ' \n')
# print(lines.sample(10))

# 문자 집합 구축
src_vocab = set()
for line in lines.src:  # 1줄씩 읽음
    for char in line:  # 1개의 문자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab_size = len(src_vocab) + 1
tar_vocab_size = len(tar_vocab) + 1
# print('source 문장의 char 집합 :', src_vocab_size)
# print('target 문장의 char 집합 :', tar_vocab_size)

# 각 문자에 index 부여.
src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))

src_to_index = dict([(word, i + 1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i + 1) for i, word in enumerate(tar_vocab)])
# print(src_to_index)
# print(tar_to_index)

# encoder 정수 인코딩 수행.
encoder_input = []

# 1개의 문장
for line in lines.src:
    encoded_line = []
    # 각 줄에서 1개의 char
    for char in line:
        # 각 char을 정수로 변환
        encoded_line.append(src_to_index[char])
    encoder_input.append(encoded_line)

# print('source 문장의 정수 인코딩 :', encoder_input[:5])

# decoder 정수 인코딩 수행.
decoder_input = []
for line in lines.tar:
    encoded_line = []
    for char in line:
        encoded_line.append(tar_to_index[char])
    decoder_input.append(encoded_line)
# print('target 문장의 정수 인코딩 :', decoder_input[:5])

# decoder의 예측값과 비교하기 위한 실제값.
# 실제 값에는 <sos> 가 있을 필요가 없으니 \t 제거.

decoder_target = []
for line in lines.tar:
    timestep = 0
    encoded_line = []
    for char in line:
        if timestep > 0:
            encoded_line.append(tar_to_index[char])
        timestep = timestep + 1
    decoder_target.append(encoded_line)
# print('target 문장 레이블의 정수 인코딩 :', decoder_target[:5])

# 패딩을 맞춰주기 위해서 src 와 tar 의 최대 길이를 구한다.

max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
# print('source 문장의 최대 길이 :',max_src_len)
# print('target 문장의 최대 길이 :',max_tar_len)


# 패딩을 줘서 src 끼리는 모든 문장의 길이를 맞춰주고, tar 끼리 모든 문장의 길이를 맞춰줌.

encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

# 모든 값에 대해 one-hot encoding 진행. 문자 단위 번역기이므로 word embedding 은 필요 하지 않음.
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

# Data Preprocessing finish
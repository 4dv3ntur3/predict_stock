#2020-11-20 과제
#2020-11-23 삼성전자 시가 맞추기


import numpy as np
import pandas as pd

size = 5

#data가 2차원일 경우
def split_x(seq, size):
    aaa = [] #는 테스트

    # seq = dataset.shape[0]

    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size), :]
        aaa.append(subset)
               
    return np.array(aaa)



#########*.npy
#*.npy 불러와서 전처리, 모델 구성, 컴파일/훈련, 평가/예측

#1_1. samsung
samsung = np.load('./data/samsung.npy', allow_pickle=True).astype('float32')
bit = np.load('./data/bit.npy', allow_pickle=True).astype('float32')
gold = np.load('./data/gold.npy', allow_pickle=True).astype('float32')
kos = np.load('./data/kos.npy', allow_pickle=True).astype('float32')



#samsung_X, Y 분리
samsung_x = split_x(samsung, size)
samsung_y = split_x(samsung, size+1)

samsung_x = samsung_x[:samsung_x.shape[0], :, :size-1]
samsung_y = samsung_y[:samsung_x.shape[0], size, size-1:]


#bit_X, Y 분리
bit_x = split_x(bit, size)
bit_y = split_x(bit, size+1)

bit_x = bit_x[1:samsung_y.shape[0], :, :size-1]
bit_y = bit_y[1:samsung_y.shape[0], size, size-1:]


#gold_X, Y 분리
gold_x = split_x(gold, size)
gold_y = split_x(gold, size+1)

gold_x = gold_x[1:samsung_y.shape[0], :, :size-1]
gold_y = gold_y[1:samsung_y.shape[0], size, size-1:]


#kos_X, Y 분리
kos_x = split_x(kos, size)
kos_y = split_x(kos, size+1)

kos_x = kos_x[1:samsung_y.shape[0], :, :size-1]
kos_y = kos_y[1:samsung_y.shape[0], size, size-1:]


#predict 
bit_x_predict = bit_x[-1:, :, :]
gold_x_predict = gold_x[-1:, :, :]
kos_x_predict = kos_x[-1:, :, :]

samsung_x_predict = samsung_x[-1:, :, :]
samsung_x = samsung_x[:-2, :, :]
samsung_y = samsung_y[:-1, :]


#train / test
from sklearn.model_selection import train_test_split

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(
    samsung_x, samsung_y, train_size=0.7
)

bit_x_train, bit_x_test = train_test_split(
    bit_x, train_size=0.7
)

gold_x_train, gold_x_test = train_test_split(
    gold_x, train_size=0.7
)

kos_x_train, kos_x_test = train_test_split(
    kos_x, train_size=0.7
)


#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model = load_model('./save/samsung_63950.h5')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

result = model.evaluate(
    [samsung_x_test, bit_x_test, gold_x_test, kos_x_test],
    samsung_y_test,
    batch_size=32)

print("=======model 저장=========")
y_pred_2 = model.predict(
    [samsung_x_predict, bit_x_predict, gold_x_predict, kos_x_predict])

print("loss: ", result[0])
print("2020/11/23 삼성전자 시가: ", y_pred_2)


#2020-11-20
#삼성전자 종가 맞추기
# ~15:30
#csv는 수정 금지 (불러와서 소스에서 잘라낼 것)
#잘라낸 걸 npy로 저장해서 로드?
#최소 2년 이상 사용
#각 data column 수는 다르게 (단, 최소 4개 이상 사용)


######## *.csv
#데이터

import numpy as np
import pandas as pd


#########*.npy
#*.npy 불러와서 전처리, 모델 구성, 컴파일/훈련, 평가/예측



#1_1. samsung
samsung = np.load('./data/samsung.npy', allow_pickle=True)
bit = np.load('./data/bit.npy', allow_pickle=True)

# print(samsung.shape) #626, 5
# print(bit.shape) #1199, 6


# print(samsung)
# print(bit)


samsung = np.asarray(samsung).astype('float32')
bit = np.asarray(bit).astype('float32')





#데이터 정리_2
#data에 따라 달라짐


#며칠씩?
size = 5

#data가 2차원일 경우
def split_x(seq, size):
    aaa = [] #는 테스트

    # seq = dataset.shape[0]

    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size), :]

        #aaa.append 줄일 수 있음
        #소스는 간결할수록 좋다
        # aaa.append([item for item in subset])
        aaa.append(subset)
        
        
    # print(type(aaa))
    return np.array(aaa)


samsung_x = split_x(samsung, size)
samsung_y = split_x(samsung, size+1)

#마지막 한 그룹은 predict로 
samsung_x = samsung_x[:621, :, :]
samsung_x_predict = samsung_x[621:, :, :]


print(samsung_x.shape)  #622, 5, 5
print(samsung_y.shape)



# X, Y 분리
samsung_x = samsung_x[:, :, :size-1]
samsung_y = samsung_y[:, size, size-1:]

# print(samsung_x)
# print("=====")
# print(samsung_y)


#1_2. bit
#데이터 정리_2
#data에 따라 달라짐



#며칠씩?
size = 5

bit_x = split_x(bit, size)
bit_y = split_x(bit, size+1)


# X, Y 분리
bit_x = bit_x[:samsung_x.shape[0], :, :size-1]
bit_y = bit_y[:samsung_x.shape[0], size, size-1:]

# print(bit_x)
samsung_x_predict = samsung_x[-1:, :, :]
bit_x_predict = bit_x[-1:, :, :]


#train / test
from sklearn.model_selection import train_test_split

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(
    samsung_x, samsung_y, train_size=0.7
)

bit_x_train, bit_x_test = train_test_split(
    bit_x, train_size=0.7
)


#2. 모델
#import 빠뜨린 거 없이 할 것
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout

#모델 1_samsung
input1 = Input(shape=(samsung_x_train.shape[1], samsung_x_train.shape[2]))
dense1_1 = LSTM(100, activation='relu')(input1)
dense1_2 = Dense(500, activation='relu')(dense1_1)
dense1_3 = Dense(600, activation='relu')(dense1_2)
dense1_4 = Dense(200, activation='relu')(dense1_3)
dense1_5 = Dense(30, activation='relu')(dense1_4)
output1 = Dense(1)(dense1_5)
model1 = Model(inputs=input1, outputs=output1)



#모델2_bit
input2 = Input(shape=(bit_x_train.shape[1], bit_x_train.shape[2]))
dense2_1 = LSTM(40, activation='relu')(input2)
dense2_3 = Dense(256, activation='relu')(dense2_1)
dense2_4 = Dense(1024, activation='relu')(dense2_3)
dense2_5 = Dense(200, activation='relu')(dense2_4)
dense2_6 = Dense(32, activation='relu')(dense2_5)
dense2_7 = Dense(10, activation='relu')(dense2_6)
output2 = Dense(1)(dense2_7)
model2 = Model(inputs=input2, outputs=output2)


#병합 
from tensorflow.keras.layers import Concatenate, concatenate

merge1 = Concatenate()([output1, output2])
middle1 = Dense(300, activation='relu')(merge1)
middle1 = Dense(2000, activation='relu')(middle1)
output1 = Dense(800, activation='relu')(output1)
output1 = Dense(30, activation='relu')(output1)
output1 = Dense(1)(output1)

model = Model(inputs=[input1, input2], outputs=output1)


#3. 컴파일, 훈련

#early stopping
# modelpath='./model/samsung-{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# cp = ModelCheckpoint(filepath=modelpath, 
#                      monitor='val_loss', 
#                      save_best_only=True, 
#                      mode='auto'
# )


model.fit(
    [samsung_x_train, bit_x_train],
    samsung_y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=1000, batch_size=32
)

model.save_weights('./save/samsung.h5')

#4. 평가

result = model.evaluate(
    [samsung_x_test, bit_x_test],
    samsung_y_test,
    batch_size=32)


x_predict = model.predict([samsung_x_predict, bit_x_predict])
print("loss: ", result[0])
print("예측값: ", x_predict)



#2020-11-20
#삼성전자 시가 맞추기



import numpy as np
import pandas as pd


#########*.npy
#*.npy 불러와서 전처리, 모델 구성, 컴파일/훈련, 평가/예측



#1_1. samsung
samsung = np.load('./data/samsung.npy', allow_pickle=True).astype('float32')
bit = np.load('./data/bit.npy', allow_pickle=True).astype('float32')
gold = np.load('./data/gold.npy', allow_pickle=True).astype('float32')
kos = np.load('./data/kos.npy', allow_pickle=True).astype('float32')



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


#predict data
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



#scaler
from sklearn.preprocessing import MinMaxScaler

#3차원 scaling 함수
def scaler_3(train, test, predict):
    num_sample = train.shape[0]
    num_sequence = train.shape[1]
    num_feature = train.shape[2]

    scaler = MinMaxScaler()

    for i in range(num_sequence):
        scaler.partial_fit(train[:, i, :])

    results_train = []
    results_test = []
    results_predict = []

    for v in range(num_sequence):
        results_train.append(scaler.transform(train[:, v, :])).reshape(num_sample, 1, num_feature)
        results_test.append(scaler.transform(test[:, v, :])).reshape(num_sample, 1, num_feature)
        results_predict.append(scaler.transform(predict[:, v, :])).reshape(num_sample, 1, num_feature)


    train = np.concatenate(results_train, axis=1)
    test = np.concatenate(results_test, axis=1)
    predict = np.concatenate(results_predict, axis=1)

    return train, test, predict


samsung_x_train, samsung_x_test, samsung_x_predict = scaler_3(samsung_x_train, samsung_x_test, samsung_x_predict)
bit_x_train, bit_x_test, bit_x_predict = scaler_3(bit_x_train, bit_x_test, bit_x_predict)
gold_x_train, gold_x_test, gold_x_predict= scaler_3(gold_x_train, gold_x_test, gold_x_predict)
kos_x_train, kos_x_test, kos_x_predict = scaler_3(kos_x_train, kos_x_test, kos_x_predict)


print(samsung_x_train.shape)


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
# model1 = Model(inputs=input1, outputs=output1)



#모델2_bit
input2 = Input(shape=(bit_x_train.shape[1], bit_x_train.shape[2]))
dense2_1 = LSTM(40, activation='relu')(input2)
dense2_3 = Dense(256, activation='relu')(dense2_1)
dense2_4 = Dense(1024, activation='relu')(dense2_3)
dense2_5 = Dense(200, activation='relu')(dense2_4)
dense2_6 = Dense(32, activation='relu')(dense2_5)
dense2_7 = Dense(10, activation='relu')(dense2_6)
output2 = Dense(1)(dense2_7)
# model2 = Model(inputs=input2, outputs=output2)


#모델2_bit
input3 = Input(shape=(gold_x_train.shape[1], gold_x_train.shape[2]))
dense3_1 = LSTM(40, activation='relu')(input3)
dense3_3 = Dense(256, activation='relu')(dense3_1)
dense3_4 = Dense(1024, activation='relu')(dense3_3)
dense3_5 = Dense(200, activation='relu')(dense3_4)
dense3_6 = Dense(32, activation='relu')(dense3_5)
dense3_7 = Dense(10, activation='relu')(dense3_6)
output3 = Dense(1)(dense3_7)
# model3 = Model(inputs=input3, outputs=output3)

input4 = Input(shape=(kos_x_train.shape[1], kos_x_train.shape[2]))
dense4_1 = LSTM(40, activation='relu')(input4)
dense4_3 = Dense(256, activation='relu')(dense4_1)
dense4_4 = Dense(1024, activation='relu')(dense4_3)
dense4_5 = Dense(200, activation='relu')(dense4_4)
dense4_6 = Dense(32, activation='relu')(dense4_5)
dense4_7 = Dense(1, activation='relu')(dense4_6)
output4 = Dense(1)(dense4_7)


#병합 
from tensorflow.keras.layers import Concatenate, concatenate

merge1 = Concatenate()([output1, output2, output3])
middle1 = Dense(300, activation='relu')(merge1)
middle1 = Dense(2000, activation='relu')(middle1)
output1 = Dense(800, activation='relu')(output1)
output1 = Dense(30, activation='relu')(output1)
output1 = Dense(1)(output1)

model = Model(inputs=[input1, input2, input3, input4], outputs=output1)



#3. 컴파일, 훈련

#early stopping
modelpath='./model/samsung_scaler-{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

cp = ModelCheckpoint(filepath=modelpath, 
                     monitor='val_loss', 
                     save_best_only=True, 
                     mode='auto'
)


model.fit(
    [samsung_x_train, bit_x_train, gold_x_train, kos_x_train],
    samsung_y_train,
    callbacks=[early_stopping, cp],
    validation_split=0.3,
    epochs=1000, batch_size=32
)


model.save_weights('./save/samsung_scale.h5')

#4. 평가

result = model.evaluate(
    [samsung_x_test, bit_x_test, gold_x_test, kos_x_test],
    samsung_y_test,
    batch_size=10)


x_predict = model.predict([samsung_x_predict, bit_x_predict, gold_x_predict, kos_x_predict])
print("loss: ", result[0])
print("2020/11/23 삼성전자 시가: ", x_predict)

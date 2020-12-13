#2020-11-12 (4일차)
#과제: 함수 응용 -> 행만 자르지 말고, 행렬 째로 자를 수 있게 
#tensorflow 5번...


import numpy as np

dataset = np.array(range(1, 10))

#개수가 똑같아야 한다 
dataset = np.array([range(100), range(100, 200), range(500, 600)])

# dataset = np.array([range(101, 201), range(311, 411), range(100)])

# print(dataset)



dataset = dataset.T
size = 5


#data가 2차원일 경우
def split_x(dataset, size):
    aaa = [] #는 테스트
    seq = dataset.shape[0]

    print(seq)
    for i in range(seq-size+1):
        subset = dataset[i:(i+size), :]

        #aaa.append 줄일 수 있음
        #소스는 간결할수록 좋다
        # aaa.append([item for item in subset])
        aaa.append(subset)
        
        
        
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(dataset, 5)
print(dataset)

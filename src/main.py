
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense

# https://hyjykelly.tistory.com/12?category=783630
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

plt.figure()

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
X_train = X_train.reshape(-1, 32*32*3)/255
y_train = np_utils.to_categorical(y_train, 10)

X_test = X_test.reshape(-1, 32*32*3)/255
y_test = np_utils.to_categorical(y_test, 10)

# 입력 데이터 크기 : 32*32 픽셀, RGB형 이미지 데이터 (데이터 전처리 포스팅 참고)
in_size = 32*32*3

# 출력 데이터 크기 : 10개의 카테고리
num_classes = 10

# keras는 sequential 함수를 제공하는데, 층을 차례대로 쌓은 모델을 만들어준다.
model = Sequential()
# 512개의 유닛을 가진 입력 layer를 만든다. activation은 층의 활성화함수(의사결정 함수)를 설정하는 매개변수다. 대표적인 함수 중 하나인 relu를 사용했다.
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
# 10가지 카테고리 중 하나로 이미지를 분류할 것이기 때문에 10개의 유닛을 가진 출력 layer를 생성했다. 활성화함수는 softmax를 사용했다.
model.add(Dense(num_classes,activation='softmax'))

model.compile(
    # 손실함수를 지정한다. 손실함수를 최소화하는 방향으로 가중치(w)와 역치(k)를 수정하게 된다. 대표적인 손실함 수 중 하나인 categorical_crossentropy 함수를 사용했다.
    loss='categorical_crossentropy',
    # 훈련과정을 설정한다. 이 함수를 최적화하는 방향으로 학습이 일어나는 대표적인 함수 중 하나인 adam을 사용했다.
    optimizer='adam',
    # 훈련과정을 모니터링하는 방식을 지정한다. accuracy를 지정하면 학습과정에서 정확도를 수집한다.
    metrics=['accuracy']
)

hist = model.fit(X_train, y_train,
                 # MLP 모델은 데이터를 작은 배치로 나누고 훈련과정에서 순회하는 방식으로 학습을 진행한다. batch_size는 1회 계산에 사용한 데이터를 지정한다.
                 batch_size=32,
                 # 에포크 = 1 은 전체 입력 데이터를 한번 순회한다는 것을 의미한다. 50으로 지정해 50번 순회하도록 했다.
                 epochs=50,
                 verbose=1,
                 # 에포크가 한번 끝날 때마다 검증데이터를 통해 데이터의 손실과 측정 지표를 출력한다. 검증데이터로 쓸 데이터를 지정한다.
                 validation_data=(X_test, y_test))

# evaluate 함수를 사용하면 모델의 최종적인 정답률과 loss 값을 알 수 있다.
# loss는 예측값과 실제값이 차이나는 정도를 나타내는 지표이다.
#
# 작을 수록 좋다.
score = model.evaluate(X_test, y_test, verbose=1)
print('정답률 = ', score[1], 'loss=', score[0])


print(hist.history)
# train set의 지표
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('Accuracy')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# validation set의 지표
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save_weights('cifar10-weight.h5')
# for i in range(0, 40):
#     im = Image.fromarray(X_train[i])
#     plt.subplot(5, 8, i+1)
#     plt.title(labels[y_train[i][0]])
#     plt.tick_params(labelbottom="off", bottom="off")
#     plt.tick_params(labelleft="off", left="off")
#     plt.imshow(im)
#
# plt.show()

!pip install pillow

from google.colab import drive

drive.mount("/content/drive")
%cd /content/drive/'My Drive'/MachineLearning/ETL8G
%ls -a

import numpy as np
import scipy.misc
from keras import backend as K
from keras import initializers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

nb_classes = 72
# input image dimensions
img_rows, img_cols = 32, 32
# img_rows, img_cols = 127, 128
X_train = np.load("hiragana.npz")['arr_0'].reshape([-1, 32, 32]).astype(np.float32) / 15
Y_train = np.repeat(np.arange(nb_classes), 160)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# カテゴリラベルをバイナリのダミー変数に変換する
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

model = Sequential()

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

print(X_train.shape)
print(Y_train.shape)
# モデルの構築
model = Sequential()

# MNISTはチャンネル情報を持っていないので畳み込み層に入れるため追加する
# model.add(Reshape((32,32,1),input_shape=(32,32)))
model.add(Conv2D(32,(3,3))) # 畳み込み層1
model.add(Activation("relu"))
model.add(Conv2D(32,(3,3))) # 畳み込み層2
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2))) # プーリング層
model.add(Dropout(0.5))

model.add(Conv2D(64,(3,3))) # 畳み込み層3
model.add(Activation("relu"))
model.add(Conv2D(64,(3,3))) # 畳み込み層4
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256)) # 全結合層
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(nb_classes)) # 出力層
model.add(Activation("softmax"))

# モデルのコンパイル
# 損失関数：交差エントロピー、最適化関数：sgd、評価関数：正解率(acc)
# model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# # 学習。バッチサイズ:200、ログ出力：プログレスバー、反復数：50、検証データの割合：0.1
# hist = model.fit(X_train, Y_train, batch_size=16, verbose=1, epochs=50, validation_split=0.1)


model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=16, verbose=1, epochs=400, validation_split=0.1)


# 学習結果の評価。今回は正答率(acc)
score = model.evaluate(X_test, Y_test, verbose=1)
print("test accuracy：", score[1])

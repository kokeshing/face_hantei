import os
import glob
from PIL import Image
import cv2
import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np

def vgg16(n):
	chara = []
	result_dir = './'

	classes = ['haruka','miki','tihaya']
	nb_classes = 3

	img_height, img_width = 200, 200
	channels = 3

	# VGG16
	input_tensor = Input(shape=(img_height, img_width, channels))
	vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

	# FC
	fc = Sequential()
	fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
	fc.add(Dense(256, activation='relu'))
	fc.add(Dropout(0.5))
	fc.add(Dense(nb_classes, activation='softmax'))

	# VGG16とFCを接続
	model = Model(input=vgg16.input, output=fc(vgg16.output))

	# 学習済みの重みをロード
	model.load_weights(os.path.join(result_dir, 'finetuning.h5'))

	model.compile(loss='categorical_crossentropy',
    	          optimizer='adam',
        	      metrics=['accuracy'])

	for i in range(n):
		# 画像を読み込んで4次元テンソルへ変換
		img = image.load_img('./face/' + str(i) + '.jpg', target_size=(img_height, img_width))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)

		# クラスを予測
		# 入力は1枚の画像なので[0]のみ
		pred = model.predict(x)[0]

		top_indices = pred.argsort()[-1:][::-1]
		chara.append([classes[top_indices[0]], pred[top_indices[0]]])
	
	return  chara

def facecut(imgpath):
	cascade_path = "./lbpcascade_animeface.xml"
	image = cv2.imread(imgpath,1)
	
	cascade = cv2.CascadeClassifier(cascade_path)
	facerect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))
	i = 0

	if len(facerect) > 0:
		for rect in facerect:
		# 顔だけ切り出して保存
			x = rect[0]
			y = rect[1]
			width = rect[2]
			height = rect[3]
			dst = image[y:y + height, x:x + width]
			save_path = './face/' + str(i) + '.jpg'
			#認識結果の保存
			cv2.imwrite(save_path, dst)
			i += 1
	
	return facerect

def resize():
	files = glob.glob('./face/*.jpg')
	for f in files:
		img = Image.open(f)
		img_resize = img.resize((200, 200), Image.LANCZOS)
		ftitle, fext = os.path.splitext(f)
		img_resize.save(ftitle + fext)



def imgpro(chara, facerect, imgpath):
	image = cv2.imread(imgpath,1)
	i = 0
	for rect in facerect:
		x = rect[0]
		y = rect[1]
		width = rect[2]
		height = rect[3]
		cv2.rectangle(image, (x, y), (x+width, y+height), (0,0,0), 10)
		cv2.putText(image, chara[i][0] + str(int(chara[i][1] * 100)) + "%", (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
		i += 1

	cv2.imwrite('./ans.jpg', image)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("usage: python hantei.py [filepath]")
		sys.exit(1)

	filepath = sys.argv[1]
	facerect = facecut(filepath)
	resize()
	chara = vgg16(len(facerect))
	imgpro(chara, facerect, filepath)
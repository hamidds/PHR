import collections

from PIL import Image
from torchvision.transforms import ToTensor, Normalize
import torch
import cv2
import numpy as np

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def resize_padding(img, width, height):
	'''

	:param img: input image
	:param width: desired width size
	:param height: desired height size
	:return: image with the new size
	'''

	desiredW, desiredH = width, height
	img_shape = img.shape
	imgW, imgH = img_shape[1], img_shape[0]
	ratio = 1.0 * imgW/imgH
	newW = int(desiredH * ratio)
	newW = newW if desiredW == None else min(desiredW, newW)
	img = cv2.resize(img, (newW, desiredH), cv2.INTER_AREA)

	# padding image
	if desiredW != None and desiredW > newW:
		white = [255, 255, 255]
		left_border = int((desiredW - newW) / 2)
		right_border = desiredW - left_border - newW
		img = cv2.copyMakeBorder(img.copy(), 0, 0, left_border, right_border, cv2.BORDER_CONSTANT, value=white)

	#img = ToTensor()(img)
	#img = Normalize(mean, std)(img)

	return img

class strLabelConverter(object):
	'''
	Convert between str and label
	NOTE:
		Insert `blank` to the words for CTC

	:param word (str): set of the possible words
	'''

	def __init__(self, words):
		self.words = words + '-'
		self.dict = {}

		for i, word in enumerate(words):
			# NOTE: 0 is reserved for `blank` required by wrap_ctc
			self.dict[word] = i + 1

	def encode(self, text):
		'''
		Support batch or single str.

		:param text (str or list of str): texts to convert.
		:return:
			torch.IntTensor [length_0 + length_1 + ... + length_{n-1}]: encoded texts.
			torch.IntTensor [n]: length of each text.
		'''

		if isinstance(text, str):
			text = [self.dict[word] for word in text]
			length = [len(text)]

		elif isinstance(text, collections.Iterable):
			length = [len(s) for s in text]
			text = ''.join(text)
			text, _ = self.encode(text)

		return torch.IntTensor(text), torch.IntTensor(length)

	def decode(self, t, length, raw=False):
		'''
		Decode encoded texts into strs

		:param t: torch.IntTensor [length_0 + length_1 + ... + length_{n-1}]: encoded texts.
		:param length: torch.IntTensor [n]: length of each text

		:raises
			AssertionError: when the texts and its length does not match.

		:return:
			text (str or list of str): texts to convert.
		'''

		if length.numel() == 1:
			length = length[0]
			assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)



#if __name__ == '__main__':

	#path_to_image = '../data/images/10000.jpg'
	#img = cv2.imread(path_to_image)


	#img = resize_padding(img, 500, 50)
	#cv2.imshow('image', img)
	#cv2.waitKey(0)
import collections

from PIL import Image
from torchvision.transforms import ToTensor, Normalize
import torch

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
	imgW, imgH = img.size
	ratio = 1.0 * imgW/imgH
	newW = int(desiredH * ratio)
	newW = newW if desiredW == None else min(desiredW, newW)
	img = img.resize((newW, desiredH), Image.ANTIALIAS)

	# padding image
	if desiredW != None and desiredW > newW:
		new_img = Image.new("RGB", (desiredW, desiredH), color=255)
		new_img.paste(img, (0, 0))
		img = new_img

	img = ToTensor()(img)
	img = Normalize(mean, std)(img)

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

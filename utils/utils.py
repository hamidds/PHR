from PIL import Image
from torchvision.transforms import ToTensor, Normalize

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
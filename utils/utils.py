import os
import num2word
import cv2
import random

words_path = "../data/words/"

def hconcat_resize(img_list):

	# take minimum height
	h_max = max(img.shape[0] for img in img_list)

	white=[255,255,255]
	img_list_resize = [cv2.copyMakeBorder(img, int((h_max-img.shape[0])/2), h_max-img.shape[0]-int((h_max-img.shape[0])/2), 0, 0, cv2.BORDER_CONSTANT, value=white)
					   for img in img_list]

	# return final image
	return cv2.hconcat(img_list_resize)

def display_num_image(num_string):

	words = num_string.split(" ")
	word_img_list = []
	for word in words:

		image_dir_path = words_path + word + "/"
		all_images_in_directory = os.listdir(image_dir_path)
		# choose a random word image from the related directory
		random_word_image_path = random.choice(all_images_in_directory)
		word_img = cv2.imread(image_dir_path + random_word_image_path)
		# concat word images horizontally
		word_img_list.insert(0, word_img)

	num_string_image = hconcat_resize(word_img_list)
	cv2.imshow('test.jpg', num_string_image)
	cv2.waitKey()



if __name__ == '__main__':

	random_num = 123120000
	num_string = num2word.convert(random_num)
	num_string += " ریال"

	print(num_string)
	display_num_image(num_string)

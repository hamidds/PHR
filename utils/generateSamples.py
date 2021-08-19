'''
This script is for generating samples using combining words images
which exist in /words directory.
'''

import os
import random

import num2word
import randomNumGenerator
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

words_path = "../data/words/"


def hconcat_resize(img_list):
    # take minimum height
    h_max = max(img.shape[0] for img in img_list)
    white=[255,255,255]
    img_list_resize = [cv2.copyMakeBorder(img, int((h_max-img.shape[0])/2), h_max-img.shape[0]-int((h_max-img.shape[0])/2), 0, 0, cv2.BORDER_CONSTANT, value=white)
					   for img in img_list]

    # return final image
    return cv2.hconcat(img_list_resize)


def create_num_image(num_string):
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
    return num_string_image

def generate_images(images_dir='../data/images/', labels_dir='../data/labels/'):
    # create the /images and /labels directory in case they do not exist
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    # fetch the random numbers generated via randomNumGenerator.py script
    random_numbers = randomNumGenerator.generate_random_num()
    random.shuffle(random_numbers)

    # create the images
    print('Creating the images...')
    idx = 0
    for num in random_numbers:
        print('{0}/{1}: {2}'.format(idx, len(random_numbers), num))
        num_string = num2word.convert(num)
        num_string += " ریال"
        num_img = create_num_image(num_string)

        # save the image and label into /images and /labels directory respectively
        cv2.imwrite(images_dir + str(num) + '.jpg', num_img)
        f = open(labels_dir + str(num) + '.txt', 'w')
        f.write(num_string)
        f.close()

        idx += 1

    # create train test list text file
    random_numbers = np.array(random_numbers)
    train_samples, test_samples = train_test_split(random_numbers, test_size=0.1)

    print('Creating train.txt and test.txt list files...')
    f = open('../data/train.txt', 'w')
    for sample in train_samples:
        f.write(str(sample) + '.jpg\n')

    f = open('../data/test.txt', 'w')
    for sample in test_samples:
        f.write(str(sample) + '.jpg\n')

    f.close()

if __name__ == '__main__':
    generate_images(images_dir='../data/images/', labels_dir='../data/labels/')
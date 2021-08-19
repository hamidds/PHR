import os

def generate_words_list(words_path='../data/words'):
    f = open('../data/words.txt', 'w')
    for word in os.listdir(words_path):
        if not word == '.DS_Store':
            f.write(word + '\n')
    f.close()

if __name__ == "__main__":
    generate_words_list(words_path='../data/words')
"""
    Attributions:
    MNIST data loader from https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
"""
import matplotlib
import numpy as np
from preprocess_input import join, MnistDataloader
from neural_net import NN
import random
import matplotlib.pyplot as plt


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 4
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 30))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image)
        if title_text != '':
            plt.title(title_text, fontsize=30)
        index += 1
    plt.ioff()
    plt.show()


#
# Show some random training and test images
#
def display_input(x_train, y_train, x_test, y_test):
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

    show_images(images_2_show, titles_2_show)


def main():
    # Paths :(
    input_path = input("Path to MNIST archive folder:\t")
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte\\train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte\\train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte\\t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    display_input(x_train, y_train, x_test, y_test)

    #NN(x_train, y_train, 5000, 0.00001, 3, 10)


if __name__ == '__main__':
    main()

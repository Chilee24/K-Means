import matplotlib.pyplot as plt
import numpy
from sklearn.cluster import KMeans


def kmeans_img(file_name, num_color):
    img = plt.imread(f"test/{file_name}")
    width = img.shape[0]
    height = img.shape[1]
    img = img.reshape(height * width, 3)
    kmeans = KMeans(n_clusters=num_color).fit(img)
    labels = kmeans.predict(img)
    cluster = kmeans.cluster_centers_
    print(cluster)
    print(labels)

    img2 = numpy.zeros_like(img)
    for i in range(len(img2)):
        img2[i] = cluster[labels[i]]

    img2 = img2.reshape(width, height, 3)
    plt.imshow(img2)
    plt.show()


if __name__ == '__main__':
    kmeans_img("bird.jpg", 50)

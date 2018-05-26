# test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
'''KMeans(algorithm='auto', copy_x=True,
            init='k-means++', max_iter=300,
            n_clusters=4, n_init=10,
           n_jobs=1, precompute_distances='auto',
           random_state=None, tol=0.0001,
          verbose=0)
return
cluster_centers_ : array, [n_clusters, n_features]  Coordinates of cluster centers
labels_ : :  Labels of each point
inertia_ : float  Sum of squared distances of samples to their closest cluster center'''
from sklearn.metrics import pairwise_distances_argmin
'''Compute minimum distances between one point and a set of points.计算一个点和一组点之间的最小距离。
This function computes for each row in X, the index of the row of Y which is closest (according to the specified distance).
This is mostly equivalent to calling:
pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis)
but uses much less memory, and is faster for large arrays.#但使用更少的内存,对大型数组更快。
This function works with dense 2D arrays only.'''#function只对二维数组有效
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 4#kg

# Load the Summer Palace photo
china = load_sample_image("china.jpg")

'''Convert to floats instead of the default 8 bits integer coding.
Dividing by 255 is important so that plt.imshow behaves works well on float data(need to be in the range [0-1])'''
china = np.array(china, dtype=np.float64) / 255 #转换为浮点数而不是默认的8位整数编码。除以255为了让imshow浮动数据在范围[0-1]上运行

# Load Image and transform to a 2D numpy array. #是特征工程吗？
w, h, d = original_shape = tuple(china.shape)#1440 1080 3
image_array = np.reshape(china, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array)[:1000]#kg

kmeans = KMeans(n_clusters=n_colors).fit(image_array_sample)#color分为n个cluster

print("done in %0.3fs." % (time() - t0))

# Get labels for all points #得到所有点的聚类标签
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)#[3 3 3 ... 1 1 1]

print("done in %0.3fs." % (time() - t0))

codebook_random = shuffle(image_array)[:n_colors]#kg, random_state=0,删去色调可变 n_colors+ 1（常数）

print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)

print("done in %0.3fs." % (time() - t0))

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""#从book和labels中重新创建(压缩的)图像。
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image
    
# Display all results, alongside original image #显示所有结果，与原始图像。
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))#labels[2 2 2 ... 0 0 0]

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()

import numpy as np
import PIL.Image as Image
import os

def centralization(images, d=320*243):
    '''
        images是图片集合（矩阵集合）
        d是图片拉伸后的维度
        输出中心化后的矩阵central_vectors和平均值
    '''
    n = len(images) # n是图片个数
    central_vectors = np.zeros([d, n])
    for i in range(n):
        images[i] = images[i].flatten()
        central_vectors[:, i] = images[i]

    row_mean = np.mean(central_vectors, axis=1).reshape(d, 1)
    central_vectors = central_vectors - row_mean
    return central_vectors, row_mean.reshape(d)


def PCA(W):
    [xshape, yshape] = W.shape
    if (xshape <= yshape):
        cov_matrix = np.dot(W, W.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(cov_matrix)
    else:
        cov_matrix = np.dot(W.T, W)
        [eigenvalues, eigenvectors] = np.linalg.eigh(cov_matrix)
        eigenvectors = np.dot(W, eigenvectors)
        for i in range(yshape):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    # 排序特征值
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[: , idx]

    eigenvalues_sum = np.sum(eigenvalues)
    tmp = 0
    num_components = 0
    # 选择合适的主成分数量
    for i in range(yshape):
        if(tmp / eigenvalues_sum > 0.95):
            break
        else:
            tmp = tmp + eigenvalues[i]
            num_components = num_components + 1

    return eigenvalues[0:num_components], eigenvectors[:,0:num_components]


# 用特征向量重构X
def reconstrcut(eigenvectors, X):
    return np.dot(np.dot(X, eigenvectors), eigenvectors.T) 


def load_images(path, height = 320, width = 243):
    files = os.listdir(path)
    ims = []
    labels = []
    for filename in files:
        try:
            im = Image.open(os.path.join(path, filename)).\
                resize((height, width), Image.ANTIALIAS).convert("L")
            ims.append(np.asarray(im, dtype=np.uint8))
            labels.append(filename[7:9]) # 文件名的7到9位是标签
        except:
            continue
    return ims, labels, files
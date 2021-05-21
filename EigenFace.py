import numpy as np
import PIL.Image as Image
import os

def centralization(images, d=320*243):
    '''
        images是图片集合（矩阵集合）
        d是图片拉伸后的维度
        输出中心化后的矩阵central_vectors
    '''
    n = len(images) # n是图片个数
    central_vectors = np.zeros([d, n])
    for i in range(n):
        images[i] = images[i].flatten()
        central_vectors[:, i] = images[i]

    row_mean = np.mean(central_vectors, axis=1).reshape(d, 1)
    central_vectors = central_vectors - row_mean
    return central_vectors, row_mean.reshape(d)

def PCA(W, num_components = 20):
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

    return eigenvalues[0:num_components], eigenvectors[:,0:num_components]


def load_images(path, height = 320, width = 243):
    files = os.listdir(path)
    ims = []
    labels = []
    for filename in files:
        try:
            im = Image.open(os.path.join(path, filename)).\
                resize((height, width), Image.ANTIALIAS).convert("L")
            ims.append(np.asarray(im, dtype=np.uint8))
            labels.append(filename[7:9])
        except:
            continue
    return ims, labels

# 用特征向量重构X
def reconstrcut(eigenvectors, X):
    return np.dot(np.dot(X, eigenvectors),eigenvectors.T) 


def predict(path, eigen_dic, meanface_dic):
    images, labels = load_images(path)
    predict_result = []
    for image in images:
        image = image.flatten()
        tmp = []
        for key in eigen_dic:
            image_tmp = image - meanface_dic[key]
            image_rec = reconstrcut(eigen_dic[key][1], image_tmp)
            distance = np.sqrt(np.sum((image_rec - image_tmp)**2))
            tmp.append(distance)
        
        idx = np.argmin(tmp)
        label = list(eigen_dic.keys())[idx]
        predict_result.append(label)

    return predict_result


# B = [np.array([[1,3],[6,4]]), np.array([[2,6],[5,8]])]
# A = centralization(B, 4, 2)
# a, b = PCA(A, 1)
train_path = './yalefaces/train'
test_path = './yalefaces/test'
images, labels = load_images(train_path)

image_dic = {}
eigen_dic = {}
meanface_dic = {}
for i in range(len(labels)):
    if i == 0 or labels[i] != labels[i-1]:
        image_dic[labels[i]] = [images[i]]
    else:
        image_dic[labels[i]].append(images[i])

for key in image_dic:
    W, row_mean = centralization(image_dic[key])
    eigenvalues, eigenvectors = PCA(W)
    eigen_dic[key] = [eigenvalues, eigenvectors]
    meanface_dic[key] = row_mean

predict_result = predict(test_path, eigen_dic, meanface_dic)
print(' ')
import numpy as np
from utils import centralization, PCA, load_images, reconstrcut


def predict(path, eigen_dic, meanface_dic):
    images, labels = load_images(path)
    predict_result = []
    for image in images:
        
        image = image.flatten()
        tmp = []

        # 用所有类别的特征向量重构该图片
        for key in eigen_dic:
            image_tmp = image - meanface_dic[key]
            image_rec = reconstrcut(eigen_dic[key][1], image_tmp)
            distance = np.sqrt(np.sum((image_rec - image_tmp)**2))
            tmp.append(distance)
        
        idx = np.argmin(tmp) # 选出重构误差最小的
        label = list(eigen_dic.keys())[idx]
        predict_result.append(label)

    correct = 0
    for i in range(len(predict_result)):
        if(labels[i] == predict_result[i]):
            correct = correct + 1
    
    print('预测正确率：{}'.format(correct / len(predict_result)))


if __name__ == '__main__':
    train_path = './yalefaces/train'
    test_path = './yalefaces/test'
    images, labels = load_images(train_path)

    image_dic = {}
    eigen_dic = {} # 特征脸字典
    meanface_dic = {} # 平均脸字典

    for i in range(len(labels)):
        if i == 0 or labels[i] != labels[i-1]:
            image_dic[labels[i]] = [images[i]]
        else:
            image_dic[labels[i]].append(images[i])

    for key in image_dic:
        W, mean_face = centralization(image_dic[key])
        eigenvalues, eigenvectors = PCA(W)
        eigen_dic[key] = [eigenvalues, eigenvectors]
        meanface_dic[key] = mean_face

    predict(test_path, eigen_dic, meanface_dic)

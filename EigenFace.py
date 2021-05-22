import numpy as np
from utils import centralization, PCA, load_images, reconstrcut


def predict(path, eigen_dic, meanface_dic):
    images, labels, files = load_images(path)
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
        predict_label = list(eigen_dic.keys())[idx]
        predict_result.append(predict_label)
    
    return predict_result, labels, files


def accuracy(predict_result, labels):
    correct = 0
    for i in range(len(predict_result)):
        if(labels[i] == predict_result[i]):
            correct = correct + 1
    
    print('预测正确率：{}'.format(correct / len(predict_result)))
    print(' ')
    return correct / len(predict_result)

if __name__ == '__main__':
    train_path = './yalefaces/train'
    test_path = './yalefaces/test'

    ap = 0
    k = 10
    for train_id in range(k):
        images, labels, _ = load_images(train_path + str(train_id))

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

        predict_result, labels, files = predict(test_path + str(train_id), eigen_dic, meanface_dic)
        print('训练集：train{}，测试集：test{}'.format(train_id, train_id))
        ap = ap + accuracy(predict_result, labels)

        with open('predict_result' + str(train_id) + '.txt','w') as f:
            f.write('训练集：train{}，测试集：test{}\n'.format(train_id, train_id))
            for i in range(len(files)):
                f.write('文件{}，预测{}，真实{}\n'.format(files[i], predict_result[i], labels[i]))

    print('{}折交叉验证平均精度{}，详情请参见生成的文件predict_result'.format(k, ap / k ))
import numpy as np
from scipy import stats
from utils import centralization, PCA, load_images, reconstrcut, project


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
    k = 10 # 10折
    for train_id in range(k):
        train_images, all_labels, _ = load_images(train_path + str(train_id))

        # 使用所有的图片构造特征脸
        W, mean_face = centralization(train_images)
        eigenvalues, eigenvectors = PCA(W)

        images_reconstruct = []  # 每张图片都投影到特征向量上，降维
        for i in range(W.shape[1]):
            images_reconstruct.append(project(eigenvectors, W[:, i]))

        test_images, test_labels, files = load_images(test_path + str(train_id))
        predict_result = []

        '''
            knn算法
            测试图片与训练图片计算欧式距离，
            k = 1 时，距离最小的一张图片的类别就是测试图片的类别
        '''
        KNN = 1 # KNN算法的K
        for image in test_images:
            image = image.flatten()
            image_tmp = image - mean_face
            image_rec = project(eigenvectors, image_tmp)

            distance = [] # 计算目标图片与所有图片的距离
            for i in range(len(images_reconstruct)):
                distance.append([np.sqrt(np.sum((image_rec - images_reconstruct[i])**2)), all_labels[i]])
                
            distance.sort(key=lambda x: x[0])
            output = [x[1] for x in distance[:KNN]]
            #返回前k个出现次数最多的类别
            predict_result.append(stats.mode(output)[0][0])

        print('训练集：train{}，测试集：test{}'.format(train_id, train_id))
        ap = ap + accuracy(predict_result, test_labels)

        with open('predict_result' + str(train_id) + '.txt', 'w') as f:
            f.write('训练集：train{}，测试集：test{}\n'.format(train_id, train_id))
            for i in range(len(files)):
                f.write('文件{}，预测{}，真实{}\n'.format(files[i], predict_result[i], test_labels[i]))

    print('{}折交叉验证平均精度{}，详情请参见生成的文件predict_result'.format(k, ap / k))

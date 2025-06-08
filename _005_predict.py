import os

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

from _001_split_data import get_dataset_list, get_train_and_val
from _003_model import AlexNet

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def predict():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = 4

    # 数据组织方式
    data_transform = transforms.Compose(
        [transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 加载图片
    image_path = r'AlexNet_pytorch_code/car_datas/blue/blue0001.jpg'    # 测试一张图片
    print(image_path)
    assert os.path.exists(image_path), f'file: {image_path} does not exist'
    img = Image.open(image_path)    # 打开图片，数据格式为RGB
    plt.imshow(img)     # 显示图片

    # 将图片转换为[N, C, H, W]数据组织格式
    img = data_transform(img)
    # 展开批次维度
    # unsqueeze()在dim维插入一个维度为1的维，例如原来x是n×m维的，torch.unqueeze(x,0)这返回1×n×m的tensor
    img = torch.unsqueeze(img, dim=0)

    # 读取类别标签
    json_path = 'classes.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_json = json.load(f)

    # 载入训练模型
    model = AlexNet(num_classes=num_classes).to(device)

    # 载入模型权重
    weights_path = r'AlexNet.pth'
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    # 开始用模型预测
    model.eval()
    with torch.no_grad():   # 梯度清零
        # 预测图片类别
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        # 预测结果可能性做大的结果的下标
        predict_index = torch.argmax(predict).numpy()

    img_title = f'class: {class_json[str(predict_index)]}   prob: {predict[predict_index].numpy():.3}'
    plt.title(img_title)

    for i in range(len(predict)):   # 打印预测各类别的可能性
        print(f'class: {class_json[str(i)]}   prob: {predict[i].numpy():.3}')

    # 打印图片和结果
    plt.show()


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = 4

    # 数据组织方式
    data_transform = transforms.Compose(
        [transforms.Resize((227, 227)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 加载图片
    test_list = get_dataset_list(r'test.txt')
    img_list, labels = test_list  # 获取图像路径和标签
    # 载入训练模型
    model = AlexNet(num_classes=num_classes).to(device)

    predict_labels = []
    for i in range(len(img_list)):
        image_path = f'{img_list[i]}'  # 读取一张图片
        # print(image_path)
        assert os.path.exists(image_path), f'file: {image_path} does not exist'
        img = Image.open(image_path)  # 打开图片，数据格式为RGB

        # 将图片转换为[N, C, H, W]数据组织格式
        img = data_transform(img)
        # 展开批次维度
        # unsqueeze()在dim维插入一个维度为1的维，例如原来x是n×m维的，torch.unqueeze(x,0)这返回1×n×m的tensor
        img = torch.unsqueeze(img, dim=0)

        # 读取类别标签
        json_path = 'classes.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            class_json = json.load(f)


        # 载入模型权重
        weights_path = r'AlexNet.pth'
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path))

        # 开始用模型预测
        model.eval()
        with torch.no_grad():  # 梯度清零
            # 预测图片类别
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            # 预测结果可能性做大的结果的下标
            predict_index = torch.argmax(predict).numpy()

        img_title = f'[Image {(i+1)}]\tclass:{class_json[str(predict_index)]}   prob:{predict[predict_index].numpy():.3}'
        predict_labels.append(class_json[str(predict_index)])   # 将预测结果加入预测标签列表
        print(img_title)


    # 获取测试集的标签列表
    classes_name = list(set(labels))
    classes_name.sort()
    print(f'classes_name: {classes_name}')
    print(f'labels: {labels}')
    print(f'predict_labels: {predict_labels}')

    # 统计预测准确率
    accuracy = accuracy_score(labels, predict_labels)
    print(f'accuracy: {accuracy}')

    # # 计算混淆矩阵
    Array = confusion_matrix(labels, predict_labels)
    print(f'confusion_matrix:\n{Array}')

    # 计算precision, recall, F1-score, support
    result = classification_report(labels, predict_labels, target_names=classes_name)
    print(result)

    # 保存混淆矩阵结果
    save_result_path = r'test_result.txt'
    with open(save_result_path, 'w') as f:
        f.write(f'{classes_name}\n')
        f.write(str(Array) + '\n')
        f.write(result)

if __name__ == '__main__':
    test()
    predict()

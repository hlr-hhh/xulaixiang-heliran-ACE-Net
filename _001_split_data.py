'''
    该程序适合三级目录，将存储格式为：数据集名称/类别名称/图片
    得到图片路径为： img_path = 数据集名称/类别名称/图片名
    使用img_path.split('/')得到：
        img_path[0] = 数据集名称
        img_path[1] = 类别名称
        img_path[2] = 图片名
'''
import json
import os
import matplotlib.pyplot as plt
import random

# 将数据集图片路径保存到txt文件
def write_dataset2txt(dataset_path, save_path):
    '''
    :param save_path: txt文件保存的目标路径
    :return:
    '''
    # 分类文件夹名称
    classes_name = os.listdir(dataset_path)  # 列表形式存储
    print(f'classes_name: {classes_name}')

    for classes in classes_name:
        cls_path = f'{dataset_path}/{classes}'
        for i in os.listdir(cls_path):
            img_path = f'{cls_path}/{i}'
            with open(os.path.join(save_path), "a+", encoding="utf-8", errors="ignore") as f:
                f.write(img_path + '\n')
    print('Writing dataset to file is finish!')

# 逐行读取dataset.txt
def read_dataset(read_path):
    '''
    读取数据集所有图片路径

    :param read_path:   dataset.txt文件所在路径
    :return:            返回所有图像存储路径的列表
    '''
    with open(os.path.join(read_path), "r+", encoding="utf-8", errors="ignore") as f:
        img_list = f.read().split('\n')
        img_list.remove('')     # 因为写入程序最后一个循环会有换行，所以最后一个元素是空元素，故删去
        # print(f'Read total of images: {len(img_list)}')
        random.seed(0)

        return img_list

# 读取训练集和验证集txt文件，获得图片存储路径和图片对应标签
def get_dataset_list(read_path):
    '''
    读取训练集和验证集txt文件，获得图片存储路径和图片对应标签

    :param read_path:   txt文件读取的目标路径
    :return:            返回所有图像存储路径和对应标签的列表的列表
    '''
    with open(os.path.join(read_path), "r+", encoding="utf-8", errors="ignore") as f:
        # 图片路径
        data_list = f.read().split('\n')
        # print(data_list)
        # print(f'Read total of images: {len(data_list)}')
        # 对应图片标签
        img_path = []
        labels = []
        for i in range(len(data_list)):
            image = data_list[i]
            img_path.append(image)
            label = data_list[i].split('/')[1]
            labels.append(str(label))

        # print(img_path)
        return img_path, labels

# 静态划分训练集与验证集，并将训练集和验证集的图片路径和对应的标签存入txt文件中
def write_train_val_test_list(img_list, train_rate, val_rate,
                              train_save_path, val_save_path, test_save_path):
    '''
    随机划分训练集与验证集，并将训练集和验证集的图片路径和对应的标签存入txt文件中
    本方法因使用random.seed(0)语句，所以本方法是静态划分数据集，若想实现动态划分，可注释掉random.seed(0)语句

    :param img_list:            保存图像路径的列表
    :param train_rate:          训练集数量的比率
    :param train_save_path:     训练图像保存路径
    :param val_save_path:       验证集图像保存路径
    :return:
    '''
    train_index = len(img_list) * train_rate    # 以train_index为界限，img_list[0, train_index)为训练集
    val_index = len(img_list) * (train_rate + val_rate)     # 索引在[train_index, val_index)之间的为验证集，其余的为测试集

    # 列表随机打乱顺序，放入种子数，保证随机固定，使结果可复现
    random.seed(0)
    random.shuffle(img_list)

    # 划分训练集和验证集，并写入txt文件
    for i in range(len(img_list)):
        # 写入训练集
        if i < train_index:
            with open(os.path.join(train_save_path), "a+", encoding="utf-8", errors="ignore") as f:
                if i < train_index - 1:
                    f.write(img_list[i] + '\n')
                else:
                    f.write(img_list[i])
        # 写入验证集
        elif i >= train_index and i < val_index:
            with open(os.path.join(val_save_path), 'a+', encoding='utf-8', errors='ignore') as f:
                if i < val_index - 1:
                    f.write(img_list[i] + '\n')
                else:
                    f.write(img_list[i])
        # 写入测试集
        else:
            with open(os.path.join(test_save_path), 'a+', encoding='utf-8', errors='ignore') as f:
                if i < len(img_list) - 1:
                    f.write(img_list[i] + '\n')
                else:
                    f.write(img_list[i])

    print(f'Train datasets was saved: {train_save_path}')
    print(f'Val datasets was saved: {val_save_path}')
    print(f'Test datasets was saved: {test_save_path}')
    print('Splitting datasets Finished!')

# 读取train.txt和val.txt文件中的图片路径和对应标签，并绘制柱状图
def get_train_and_val(train_txt_path, val_txt_path):
    # 读取train.txt和val.txt文件中的图片路径和对应标签
    train_img_path, train_label = get_dataset_list(train_txt_path)
    val_img_path, val_label = get_dataset_list(val_txt_path)

    # 类别的集合
    classes = list(set(train_label + val_label))   # 去重
    classes.sort()                      # 排序，固定顺序
    # 统计各类别数量
    every_class_num = []
    for cls in classes:
        # print(f'{cls} total:{train_label.count(cls) + val_label.count(cls)}')
        every_class_num.append(train_label.count(cls) + val_label.count(cls))     # 追加各类别元素的数量
    # print(every_class_num)

    # 将标签字符串转为数值
    classes_dict = {}
    for i in range(len(classes)):
        key = classes[i]
        value = i
        classes_dict[key] = value

    train_labels = []
    val_labels = []

    for label in train_label:
        train_labels.append(classes_dict[label])

    for label in val_label:
        val_labels.append(classes_dict[label])

    # 改变字典组织格式
    classes_dict = dict((v, k) for k, v in classes_dict.items())
    # 将类别写入json文件
    classes_json = json.dumps(classes_dict, indent=4)
    json_path = r'classes.json'
    with open(json_path, 'w') as f:
        f.write(classes_json)

    # 是否绘制每种类别个数柱状图
    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(classes)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(classes)), classes)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('Classes distribution')
        plt.show()

    return train_img_path, train_labels, val_img_path, val_labels, classes


if __name__ == '__main__':
    write_dataset2txt(r'car_datas', r'dataset.txt')   # 创建dataset.txt数据集，将flower_photos修改为自己的数据集名称

    img_list = read_dataset('dataset.txt')  # 读取dataset.txt中的内容
    train_rate = 0.6    # 训练集比重60%
    val_rate = 0.2      # 验证集比重20%，测试集比重20%
    train_save = r'train.txt'
    val_save = r'val.txt'
    test_save = r'test.txt'
    write_train_val_test_list(img_list, train_rate, val_rate, train_save, val_save, test_save)
    get_train_and_val('train.txt', 'val.txt')

    train_data = r'train.txt'
    val_data = r'val.txt'

    train_img_path, train_labels, val_img_path, val_labels, classes = get_train_and_val(train_data, val_data)
    print(f'classes: {classes}')

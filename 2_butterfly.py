#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !cd data/data73997 &&\
# !unzip -qo data/Butterfly20.zip
# !unzip -qo data/Butterfly20_test.zip


# # 基于PaddlePaddle2.0的蝴蝶图像识别分类——利用预训练残差网络ResNet101模型
# 
# 在陆老师的版本上做了一些修改和优化，增加了数据增加处理，以及在迭代器上使用带动量的optimizer，并对损失进行了L2权值衰减。

# ## 1. 蝴蝶识别分类任务概述
# 
# 人工智能技术的应用领域日趋广泛，新的智能应用层出不穷。本项目将利用人工智能技术来对蝴蝶图像进行分类，需要能对蝴蝶的类别、属性进行细粒度的识别分类。相关研究工作者能够根据采集到的蝴蝶图片，快速识别图中蝴蝶的种类。期望能够有助于提升蝴蝶识别工作的效率和精度。
# 

# ## 2. 创建项目和挂载数据
# 
# 数据集都来源于网络公开数据（和鲸社区）。图片中所涉及的蝴蝶总共有9个属，20个物种，文件genus.txt中描述了9个属名，species.txt描述了20个物种名。
# 
# 在创建项目时，可以为该项目挂载Butterfly20蝴蝶数据集，即便项目重启，该挂载的数据集也不会被自动清除。具体方法如下：首先采用notebook方式构建项目，项目创建框中的最下方有个数据集选项，选择“+添加数据集”。然后，弹出搜索框，在关键词栏目输入“bufferfly20”，便能够查询到该数据集。最后，选中该数据集，可以自动在项目中挂载该数据集了。
# 
# 需要注意的是，每次重新打开该项目，data文件夹下除了挂载的数据集，其他文件都将被删除。
# 
# 被挂载的数据集会自动出现在data目录之下，通常是压缩包的形式。在data/data63004目录，其中有两个压缩文件，分别是Butterfly20.zip和Butterfly20_test.zip。也可以利用下载功能把数据集下载到本地进行训练。
# 

# ## 3. 初探蝴蝶数据集
# 
# 我们看看蝴蝶图像数据长什么样子？
# 
# 首先，解压缩数据。类以下几个步骤：
# 
# 第一步，把当前路径转换到data目录，可以使用命令!cd data。在AI studio nootbook中可以使用Linux命令，需要在命令的最前面加上英文的感叹号(!)。用&&可以连接两个命令。用\号可以换行写代码。需要注意的是，每次重新打开该项目，data文件夹下除了挂载的数据集，其他文件都会被清空。因此，如果把数据保存在data目录中，每次重新启动项目时，都需要解压缩一下。如果想省事持久化保存，可以把数据保存在work目录下。
# 
# 实际上，!加某命令的模式，等价于python中的get_ipython().system('某命令')模式。
# 
# 第二步，利用unzip命令，把压缩包解压到当前路径。unzip的-q参数代表执行时不显示任何信息。unzip的-o参数代表不必先询问用户，unzip执行后覆盖原有的文件。两个参数合起来，可以写为-qo。

# In[ ]:


get_ipython().system('cd data/data73997  &&unzip -qo Butterfly20_test.zip &&unzip -qo Butterfly20.zip &&rm -r __MACOSX')


# ### 3.1 导入相关库文件

# In[ ]:


import paddle
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os 
import glob
import random
import time
import pandas as pd


print(f'Welcome to paddle  {paddle.__version__} zoo,\n there are many butterflies here today,\n please enjoy the good time with us!' )


# In[ ]:


# but_files = np.array(glob("/data/images/*/*/*"))
# print number of images in each dataset
# print('There are %d total dog images.' % len(but_files))
data_path='/home/aistudio/data/data73997/Butterfly20/*/*.jpg'
test_path='/home/aistudio/data/data73997/Butterfly20_test/*.jpg'
but_files =glob.glob(data_path)
test_files =glob.glob(test_path)


print(f'训练集样品数量为：{len(but_files)}个\n 测试集样品数量为：{len(test_files)}个')


# ### 欣赏一下各类的蝴蝶
# 每次运行均会随机产生不同的蝴蝶图片

# In[ ]:


index=random.choice(but_files)
index20 =random.sample(but_files,20)
plt.figure(figsize=(12,12),dpi=100)
for i in range(20):
    img = Image.open(index20[i])
    name=index20[i].split('/')[-2]
    plt.subplot(4, 5, i + 1)
    plt.imshow(img, 'gray')
    plt.title(name, fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.tight_layout()


# #### 随机查看一个类别
# 
# Butterfly20文件夹中有很多子文件夹，每个子文件夹下又有很多图片，每个子文件夹的名字都是蝴蝶属种的名字。由此，可以推测每个文件夹下是样本，而样本的标签就是子文件夹的名字。
# 
# 我们绘制data/Butterfly20/001.Atrophaneura_horishanus文件夹下的图片006.jpg。根据百度百科，Atrophaneura horishanus是凤蝶科、曙凤蝶属的一个物种。
# 
# 我们再绘制data/Butterfly20/002.Atrophaneura_varuna文件夹下的图片006.jpg。根据百度百科，Atrophaneura varuna对应的中文名称是“瓦曙凤蝶”，它是凤蝶科、曙凤蝶属的另一个物种。
# 
# 虽然乍一看蝴蝶都是相似的，但不同属种的蝴蝶在形状、颜色等细节方面还是存在很大的差别。

# In[ ]:


import matplotlib.pyplot as plt
import PIL.Image as Image

path='/home/aistudio/data/data73997/Butterfly20/001.Atrophaneura_horishanus/006.jpg'
img = Image.open(path)
plt.imshow(img)          #根据数组绘制图像
plt.show()               #显示图像


# In[ ]:


path='/home/aistudio/data/data73997/Butterfly20/002.Atrophaneura_varuna/006.jpg'
img = Image.open(path)
plt.imshow(img)          #根据数组绘制图像
plt.show()               #显示图像


# 更具挑战的是，即便属于同一属种，不同的蝴蝶图片在角度、明暗、背景、姿态、颜色等方面均存在不小差别。甚至有的图片里面有多只蝴蝶。以下两张蝴蝶图片均出自同一个属种Atrophaneura horishanus。

# In[ ]:


path1='/home/aistudio/data/data73997/Butterfly20/001.Atrophaneura_horishanus/006.jpg'
path2='/home/aistudio/data/data73997/Butterfly20/001.Atrophaneura_horishanus/100.jpg'


img1 = Image.open(path1)
plt.imshow(img1)          #根据数组绘制图像
plt.show()

img2 = Image.open(path2)
plt.imshow(img2)          #根据数组绘制图像
plt.show()               #显示图像


# ### 仿射填充+中心裁切增广技术,可以避免resize造成的变形

# In[ ]:


def enlarge(img): 
    h,w,_=img.shape
    ty=(600-h)//2
    tx=(600-w)//2
    # 定义平移矩阵，需要是numpy的float32类型
    # x轴平移200，y轴平移500
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    # 用仿射变换实现平移
    dst = cv2.warpAffine(img, M, (600, 600))
    dst = dst[100:501,100:501,:]
    return dst
    
index=random.choice(but_files)
#index=but_files[1]
print(index)
name=index.split('/')[-2]
img = Image.open(index)
img =cv2.imread(index)
print(img.shape)
img =img[:,:,::-1]
imgl=enlarge(img)
print(imgl.shape)
print(f'该样本标签为:{name}')
# plt.figure(figsize=(8,10),dpi=50)
# plt.axis('off')
# plt.imshow(img)   
plt.figure(figsize=(12,12))
#显示各通道信息
plt.subplot(121)
plt.imshow(img,'gray')
plt.title('RGB_Image')
plt.subplot(122)
plt.imshow(imgl,'gray')


# ## 4. 准备数据
# 
# 数据准备过程包括以下两个重点步骤：
# 
# 一是建立样本数据读取路径与样本标签之间的关系。
# 
# 二是构造读取器与数据预处理。可以写个自定义数据读取器，它继承于PaddlePaddle2.0的dataset类，在__getitem__方法中把自定义的预处理方法加载进去。
# 

# In[ ]:


# 以下代码用于建立样本数据读取路径与样本标签之间的关系
import os
import random

data_list = [] # 用个列表保存每个样本的读取路径、标签

# 由于属种名称本身是字符串，而输入模型的是数字。需要构造一个字典，把某个数字代表该属种名称。
# 键是属种名称，值是整数。
label_list=[]
with open("/home/aistudio/data/data73997/species.txt") as f:
    for line in f:
        a,b = line.strip("\n").split(" ")
        label_list.append([b, int(a)-1])
label_dic = dict(label_list)
for i in label_dic:
    print(i)

#获取Butterfly20目录下的所有子目录名称，保存进一个列表之中
class_list = os.listdir("/home/aistudio/data/data73997/Butterfly20")
class_list.remove('.DS_Store') #删掉列表中名为.DS_Store的元素，因为.DS_Store并没有样本。

for each in class_list:
    for f in os.listdir("/home/aistudio/data/data73997/Butterfly20/"+each):
        data_list.append(["/home/aistudio/data/data73997/Butterfly20/"+each+'/'+f,label_dic[each]])

#按文件顺序读取，可能造成很多属种图片存在序列相关，用random.shuffle方法把样本顺序彻底打乱。
random.shuffle(data_list)

#打印前十个，可以看出data_list列表中的每个元素是[样本读取路径, 样本标签]。
print(data_list[0:10])

#打印样本数量，一共有1866个样本。
print("样本数量是：{}".format(len(data_list)))


# ## 4.1使用pandas进行数据预处理

# In[ ]:


df = pd.DataFrame(but_files,columns=['filepath'])     #生成数据框。
df['name'] = df.filepath.apply(lambda x:x.split('/')[-2])    #按要求产生相对路径。只要工作目录下的相对路径 。
df['label']=df.name.map(label_dic) #用映射生成标签   
df['shape']=df.filepath.apply(lambda x:cv2.imread(x).shape)  #数据形状 
df['height']=df['shape'].apply(lambda x:x[0])
df['width']=df['shape'].apply(lambda x:x[1])


# ### 生成数据框，其中包含了文件的路径、样品类名、标签、数据的格式等信息

# In[ ]:


df_dataset=df[['filepath','label']]

dataset=np.array(df_dataset).tolist()
# dataset[:10]
print(len(dataset))
#  数据的最大和最小尺寸 数据的格式接近（400，600），最大尺寸为（600，600）
df.height.max(),df.width.max(),df.height.min(),df.width.min()


# ###  展示各个数据的分布情况 
# 可见数据分布并不均匀

# In[ ]:


group=df.name.value_counts() #查看样品分布情况

plt.figure(figsize=(8,4),dpi=100)
group.plot(kind='bar')


# In[ ]:


def label_suffle(df,key='label'):
    """
    数据不平衡，用上这个标签平滑 将数据分布变得均匀

    """
    label_max = df[key].value_counts().max() #获取标签数量最大值
    label_len = len(np.unique(df[key])) #获取样品标签个数
    label_balance =pd.DataFrame()
    for i in range(label_len):
        #print(len(df[df[key]==i]))
        if len(df[df[key]==i]) == label_max: #比较当前样品编号数量与最大值，如果相等则添加该样本所有数据
            label_balance=label_balance.append(df[df[key]==i])
        else:
            df_i = df[df[key]==i].sample(label_max,replace=True) #否则从该样品自身生产与最大标签数量的样本
            label_balance=label_balance.append(df_i)
    label_balance.sample(frac=1) #乱序
    return label_balance


df=label_suffle(df)
group=df.name.value_counts() #查看样品分布情况
plt.figure(figsize=(8,4),dpi=100)
group
group.plot(kind='bar')


# ### 平滑后的 训练集导出列表,操作简便快捷
# 关键的数据处理： 数据集的生成，抽离验证集后，对训练集的数据标签平滑

# In[ ]:


df = pd.DataFrame(but_files,columns=['filepath'])     #生成数据框。  
df['name'] = df.filepath.apply(lambda x:x.split('/')[-2])    #按要求产生相对路径。只要工作目录下的相对路径 。
df['label']=df.name.map(label_dic) #用映射生成标签  
del df['name']
eval_dataset=df.sample(frac=0.1)
train_dataset= df.drop(index=eval_dataset.index)
train_dataset= label_suffle(train_dataset)  # 单独对训练集的数据标签平滑

group=train_dataset.label.value_counts() #查看样品分布情况
plt.figure(figsize=(8,4),dpi=100)
group.plot(kind='bar')

train_dataset=np.array(train_dataset).tolist()
eval_dataset=np.array(eval_dataset).tolist()


# ## 5.数据预处理

# ## 根据前面数据探索得知，数据的格式在（600，600)像数范围内。
# 以下，通过opencv 的仿射，将数据填充到600*600的底片中，后续的resize，也不会照成变形。 该数据增强方式为本实践，最有意义的部分之一。

# In[ ]:


def enlarge(img): 
    h,w,_=img.shape
    ty=(600-h)//2
    tx=(600-w)//2
    # 定义平移矩阵，需要是numpy的float32类型
    # x轴平移200，y轴平移500
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    # 用仿射变换实现平移
    dst = cv2.warpAffine(img, M, (600, 600))
    dst = dst[100:501,100:501,:]
    return ds


def random_rotate(img):  
    """自定义图片随机旋转"""
    height,width,_ =img.shape
    degree=random.choice(range(0,360,10))
    size=random.uniform(0.7, 0.95)
    matRotate = cv2.getRotationMatrix2D((height*0.5, width*0.5),degree, size) # mat rotate 1 center 2 angle 3 缩放系数
    return cv2.warpAffine(img, matRotate, (width,height ))


# In[ ]:


#以下代码用于构造读取器与数据预处理
#首先需要导入相关的模块
import paddle
#from paddle.vision.transforms import Compose, ColorJitter, Resize,Transpose, Normalize，RandomRotation
from paddle.vision.transforms import Compose,CenterCrop, Resize,Normalize,RandomRotation,RandomHorizontalFlip,Transpose,ToTensor
import cv2
import numpy as np
from PIL import Image
from paddle.io import Dataset

def enlarge(img): 
    h,w,_=img.shape
    ty=(600-h)//2
    tx=(600-w)//2
    # 定义平移矩阵，需要是numpy的float32类型
    # x轴平移200，y轴平移500
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    # 用仿射变换实现平移
    dst = cv2.warpAffine(img, M, (600, 600))
    dst = dst[100:501,100:501,:]
    return dst

def random_rotate(img):
    height,width,_ =img.shape
    degree=random.choice(range(0,360,10))
    size=random.uniform(0.7, 0.95)
    matRotate = cv2.getRotationMatrix2D((height*0.5, width*0.5),degree, size) # mat rotate 1 center 2 angle 3 缩放系数
    return cv2.warpAffine(img, matRotate, (width,height ))

#自定义的数据预处理函数，输入原始图像，输出处理后的图像，可以借用paddle.vision.transforms的数据处理功能
def preprocess(img):
    transform = Compose([
        #CenterCrop(400),
        #Resize(size=(224, 224)), #把数据长宽像素调成224*224
        #ColorJitter(0.4, 0.4, 0.4, 0.4),
        RandomHorizontalFlip(0.8),
        #BrightnessTransform(0.4),
        RandomRotation((-10,10)),
        Resize(size=(224, 224)), #把数据长宽像素调成224*224
        Normalize(mean=[0, 0, 0],std=[255, 255, 255], data_format='HWC'),        
        #Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'), #标准化
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data_format='HWC'), #标准化
        Transpose(), #原始数据形状维度是HWC格式，经过Transpose，转换为CHW格式
        ])
    img = transform(img).astype("float32")
    return img


#自定义数据读取器
class Reader(Dataset):
    def __init__(self, data, is_val=False):
        super().__init__()
        #在初始化阶段，把数据集划分训练集和测试集。由于在读取前样本已经被打乱顺序，取20%的样本作为测试集，80%的样本作为训练集。
        # self.samples = data[-int(len(data)*0.2):] if is_val else data[:-int(len(data)*0.2)]
        self.samples = data

    # def __getitem__(self, idx):
    #     #处理图像
    #     img_path = self.samples[idx][0] #得到某样本的路径
    #     img = Image.open(img_path)
    #     if img.mode != 'RGB':
    #         img = img.convert('RGB')
    #     img = preprocess(img) #数据预处理--这里仅包括简单数据预处理，没有用到数据增强

    #     #处理标签
    #     label = self.samples[idx][1] #得到某样本的标签
    #     label = np.array([label], dtype="int64") #把标签数据类型转成int64
    #     return img, label
    
    def __getitem__(self, idx):
        #处理图像
        img_path = self.samples[idx][0] #得到某样本的路径
        #img = Image.open(img_path)
        img =cv2.imread(img_path)
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        img =img[:,:,::-1]
        img=enlarge(img)
        #img=random_rotate(img)
        img = preprocess(img) #数据预处理--这里仅包括简单数据预处理，没有用到数据增强
        
        #处理标签
        label = self.samples[idx][1] #得到某样本的标签
        label = np.array([label], dtype="int64") #把标签数据类型转成int64
        return img, label

    def __len__(self):
        #返回每个Epoch中图片数量
        return len(self.samples)

#生成训练数据集实例
# train_dataset = Reader(data_list, is_val=False)
train_dataset = Reader(train_dataset)
  

#生成测试数据集实例
# eval_dataset = Reader(data_list, is_val=True)
eval_dataset = Reader(eval_dataset)

#打印一个训练样本
#print(train_dataset[1136][0])
print(train_dataset[1136][0].shape)
print(train_dataset[1136][1])
print(len(train_dataset))
print(len(train_dataset),len(eval_dataset))


# In[ ]:





# ## 6. 建立模型
# 
# 为了提升探索速度，建议首先选用比较成熟的基础模型，看看基础模型所能够达到的准确度。之后再试试模型融合，准确度是否有提升。最后可以试试自己独创模型。
# 
# 为简便，这里直接采用101层的残差网络ResNet，并且采用预训练模式。为什么要采用预训练模型呢？因为通常模型参数采用随机初始化，而预训练模型参数初始值是一个比较确定的值。这个参数初始值是经历了大量任务训练而得来的，比如用CIFAR图像识别任务来训练模型，得到的参数。虽然蝴蝶识别任务和CIFAR图像识别任务是不同的，但可能存在某些机器视觉上的共性。用预训练模型可能能够较快地得到比较好的准确度。
# 
# 在PaddlePaddle2.0中，使用预训练模型只需要设定模型参数pretained=True。值得注意的是，预训练模型得出的结果类别是1000维度，要用个线性变换，把类别转化为20维度。

# In[37]:


#定义模型
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet,self).__init__()
        self.layer=paddle.vision.models.resnet50(pretrained=True)
        self.fc = paddle.nn.Linear(1000, 20)
    #网络的前向计算过程
    def forward(self,x):
        x=self.layer(x)
        
        x=self.fc(x)
        return x

mynet = MyNet()
#定义输入
input_define = paddle.static.InputSpec(shape=[-1,3,224,224], dtype="float32", name="img")
label_define = paddle.static.InputSpec(shape=[-1,1], dtype="int64", name="label")

model = paddle.Model(mynet,inputs=input_define,labels=label_define) #用Paddle.Model()对模型进行封装
model.summary((-1,3,224,224))
# network = paddle.vision.models.resnet50(num_classes=20, pretrained=True)
# model = paddle.Model(network)
# model.summary((1,3, 224, 224))


# ## 7. 应用高阶API训练模型
# 
# 一是定义输入数据形状大小和数据类型。
# 
# 二是实例化模型。如果要用高阶API，需要用Paddle.Model()对模型进行封装，如model = paddle.Model(model,inputs=input_define,labels=label_define)。
# 
# 三是定义优化器。这个使用Adam优化器，学习率设置为0.0001，优化器中的学习率(learning_rate)参数很重要。要是训练过程中得到的准确率呈震荡状态，忽大忽小，可以试试进一步把学习率调低。
# 
# 四是准备模型。这里用到高阶API，model.prepare()。
# 
# 五是训练模型。这里用到高阶API，model.fit()。参数意义详见下述代码注释。

# In[38]:


from config import * 

EPOCHS = get('epochs')
BATCH_SIZE = get('batch_size')


#实例化网络对象并定义优化器等训练逻辑


optimizer = paddle.optimizer.Adam(learning_rate=0.00375, parameters=model.parameters())
#上述优化器中的学习率(learning_rate)参数很重要。要是训练过程中得到的准确率呈震荡状态，忽大忽小，可以试试进一步把学习率调低。

def create_optim(parameters):
    # step_each_epoch = get('total_images') // get('batch_size')
    step_each_epoch = len(train_dataset)//64
    lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=get('LEARNING_RATE.params.lr'),
                                                  T_max=step_each_epoch * EPOCHS)
    # lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.00375,
    #                                               T_max=step_each_epoch * EPOCHS)

    return paddle.optimizer.Momentum(learning_rate=lr,
                                     parameters=parameters,
                                     weight_decay=paddle.regularizer.L2Decay(get('OPTIMIZER.regularizer.factor')))


model.prepare(create_optim(model.parameters()),  # 优化器
              paddle.nn.CrossEntropyLoss(),        # 损失函数
              paddle.metric.Accuracy(topk=(1, 5))) # 评估指标

# 训练可视化VisualDL工具的回调函数
visualdl = paddle.callbacks.VisualDL(log_dir='visualdl_log')



# ## 训练

# In[39]:


model.fit(train_data=train_dataset,             # 训练数据集
          eval_data=eval_dataset,               # 测试数据集
          batch_size=BATCH_SIZE,                       # 一个批次的样本数量
          epochs=EPOCHS,                            # 迭代轮次
          save_dir="/home/aistudio/gsq_train_1",  # 把模型参数、优化器参数保存至自定义的文件夹
          save_freq=2,                          # 设定每隔多少个epoch保存模型参数及优化器参数
          shuffle=True,
          verbose=1,                     
          callbacks=[visualdl]
)


# ## 8.评估
# 
# 该模型在评估数据集中精确度87.66%

# In[40]:


model.evaluate(eval_dataset,verbose=1)
model.save('butterfly', False)  # save for inference


# ## 测试数据集处理

# In[41]:


image_list =glob.glob('/home/aistudio/data/data73997/Butterfly20_test/*.jpg')
df_image=pd.DataFrame(image_list)
df_image.rename(columns={0:'file_path'}, inplace = True)
df_image['submit']=df_image.file_path.apply(lambda x:x.split('/')[-1])
df_image.sort_values(by='submit', ascending=True, inplace=True )
df_image.reset_index(drop=True)


# 测试集的预处理必须和训练集的预处理保持一致

# In[42]:


import paddle.vision.transforms as T
data_transforms = T.Compose([
    T.Resize(size=(224, 224)),   
    T.Transpose(),    # HWC -> CHW
    T.Normalize(
        mean=[0, 0, 0],        # 归一化
        std=[255, 255, 255],
        to_rgb=True)    
])


# In[44]:


paddle.set_device('gpu:0') 
# paddle.set_device('cpu')
model = paddle.jit.load("butterfly")
model.eval() #训练模式

def infer(img):
    xdata =data_transforms(Image.open(img)).reshape(-1,3,224,224)
    out = model(xdata)
    label_pre=np.argmax(out.numpy()) 
    return label_pre
infer(df_image.file_path[199])


# In[45]:


labelx=[]
for i in df_image.file_path:
    x=infer(i)
    labelx.append(x)

df_image['class_num'] = labelx
del df_image['file_path']
df_image.to_csv('submit4.csv', index=False,header=None)


# ## 测试集结果预测展示

# In[46]:


index=random.choice(image_list)
index20 =random.sample(image_list,20)
plt.figure(figsize=(12,12),dpi=100)
for i in range(20):
    img = cv2.imread(index20[i])
    name=f'predict:{infer(index20[i])}'
    plt.subplot(4, 5, i + 1)
    plt.imshow(img[:,:,::-1], 'gray')
    plt.title(name, fontsize=15,color='red')
    plt.xticks([]), plt.yticks([])
plt.tight_layout()


# ## 8.模型保存

# In[47]:


from paddle.static import InputSpec
model_2 = paddle.Model(mynet, inputs=input_define,labels=label_define)
# model_2 = paddle.Model(model,parameters=model.parameters())
visualdl = paddle.callbacks.VisualDL(log_dir='retraining_log')
#加载模型
model_2.load('/home/aistudio/gsq_train_1/final')
model_2.prepare(paddle.optimizer.Adam(learning_rate=0.000025, parameters=mynet.parameters()),
                paddle.nn.CrossEntropyLoss(),
                paddle.metric.Accuracy())

# model_2.prepare(creat_optim(model_2.parameters()),
#               paddle.nn.CrossEntropyLoss(),
#               paddle.metric.Accuracy(topk=(1,5)))

model_2.fit(train_dataset, 
            eval_dataset,
            epochs=5,
            batch_size=32,
            verbose=1,
            shuffle=True,
            save_dir='./retrain_mode/retrain',
            save_freq=2,
            callbacks=[visualdl])


# In[48]:


model_2.evaluate(eval_dataset,verbose=1)


# In[ ]:


model.save('gsq/MyNet')
model_2.save('gsq/Mynet_2')


# ## 9. 应用已经训练好的模型进行预测
# 
# 如果是要参加建模比赛，通常赛事组织方会提供待预测的数据集，我们需要利用自己构建的模型，来对待预测数据集合中的数据标签进行预测。也就是说，我们其实并不知道到其真实标签是什么，只有比赛的组织方知道真实标签，我们的模型预测结果越接近真实结果，那么分数也就越高。
# 
# 预测流程分为以下几个步骤：
# 
# 一是构建数据读取器。因为预测数据集没有标签，该读取器写法和训练数据读取器不一样，建议重新写一个类，继承于Dataset基类。
# 
# 二是实例化模型。如果要用高阶API，需要用Paddle.Model()对模型进行封装，如paddle.Model(MyNet(),inputs=input_define)，由于是预测模型，所以仅设定输入数据格式就好了。
# 
# 三是读取刚刚训练好的参数。这个保存在/home/aistudio/work目录之下，如果指定的是final则是最后一轮训练后的结果。可以指定其他轮次的结果，比如model.load('/home/aistudio/work/30')，这里用到了高阶API，model.load()
# 
# 四是准备模型。这里用到高阶API，model.prepare()。
# 
# 五是读取待预测集合中的数据，利用已经训练好的模型进行预测。
# 
# 六是结果保存

# In[ ]:


class InferDataset(Dataset):
    def __init__(self, img_path=None):
        """
        数据读取Reader(推理)
        :param img_path: 推理单张图片
        """
        super().__init__()
        if img_path:
            self.img_paths = [img_path]
        else:
            raise Exception("请指定需要预测对应图片路径")

    def __getitem__(self, index):
        # 获取图像路径
        img_path = self.img_paths[index]
        # 使用Pillow来读取图像数据并转成Numpy格式
        img = Image.open(img_path)
        if img.mode != 'RGB': 
            img = img.convert('RGB') 
        img = preprocess(img) #数据预处理--这里仅包括简单数据预处理，没有用到数据增强
        return img

    def __len__(self):
        return len(self.img_paths)

#实例化推理模型
model = paddle.Model(MyNet(),inputs=input_define)

#读取刚刚训练好的参数
model.load('gsq/MyNet')

#准备模型
model.prepare()

#得到待预测数据集中每个图像的读取路径
infer_list=[]
with open("/home/aistudio/data/testpath.txt") as file_pred:
    for line in file_pred:
        infer_list.append("/home/aistudio/data/"+line.strip())

#模型预测结果通常是个数，需要获得其对应的文字标签。这里需要建立一个字典。
def get_label_dict2():
    label_list2=[]
    with open("/home/aistudio/data/species.txt") as filess:
        for line in filess:
            a,b = line.strip("\n").split(" ")
            label_list2.append([int(a)-1, b])
    label_dic2 = dict(label_list2)
    return label_dic2

label_dict2 = get_label_dict2()
#print(label_dict2)

#利用训练好的模型进行预测
results=[]
for infer_path in infer_list:
    infer_data = InferDataset(infer_path)
    result = model.predict(test_data=infer_data)[0] #关键代码，实现预测功能
    result = paddle.to_tensor(result)
    result = np.argmax(result.numpy()) #获得最大值所在的序号
    results.append("{}".format(label_dict2[result])) #查找该序号所对应的标签名字

#把结果保存起来
with open("work/result.txt", "w") as f:
    for r in results:
        f.write("{}\n".format(r))


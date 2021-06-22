# 安装Anaconda

1）下载安装包https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

2）直接安装，在提示加入路径的时候打勾（使得conda在cmd也能生效

如果电脑中本来就有配置好的python则不要管，直接一路默认

# 创建一个新环境
存放tensorflow和它需要的python

## 检查CPU版本

从 TensorFlow 1.6 开始，二进制文件使用 AVX 指令

下载cpu-z检查电脑支不支持，支持的话下列python版本可以更换为3.8，tensorflow版本可以更换为2.x

## 创建环境

1）打开anaconda prompt

2）首先检查一下目前的环境列表#conda env list

3）创建新的环境#conda create -n env_name python=3.6

环境删除#conda remove -n env_name --all

激活环境#activate tensorflowenvs

4）检查pip版本,安装tensorflow2的话pip版本>19才行#pip -V

更新pip#pip install --upgrade pip

安装tensorflow#pip install --index-url https://pypi.douban.com/simple tensorflow==1.5.0

豆瓣下载速度真快啊（烟

## 验证

1）进入python环境#python

导入tensorflow验证#import tensorflow as tf;(报错说numpy版本是future版本)

2）关掉anconda prompt，再重新打开

激活环境#activate tensorflowenvs

安装低版本numpy#python -m pip install --index-url https://pypi.douban.com/simple numpy==1.16.0

3）重复1）步骤，导入没反应就是成功了

4）输入测试用例

<!-- a = tf.constant([1.0, 2.0], name='a')

b = tf.constant([2.0, 3.0], name='b')

result = a+b

sess=tf.Session()

print(sess.run(result)) -->

输出结果：[3. 5.]

5）退出python-退出虚拟环境

#exit（）

#conda deactivate

## 编写程序

如何在vsc中进行深度学习

1）添加anaconda的环境文件夹路径到vsc设置的python.venvPath(打开设置最上面搜索即可)

#C:/Otherapp/Anaconda/envs

可以在左下角自由切换python版本
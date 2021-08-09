## 安装步骤
* 参考https://blog.csdn.net/qq_45704942/article/details/114647667
1. 新建环境——在anaconda中新建一个环境python版本3.6
2. 安装——激活环境后在https://pytorch.org/get-started/locally/选择对应的版本，输入安装语句即可
3. 验证——prompt中输入`import torch`和`import torchvision`
4. 写代码——在VSC中写一个test.py并快捷运行/或者在prompt中分别输入`cd../..`和`Users\17853\code\Deeplearn_note\Pytorch\test.py`，如果要换盘就在中间夹上`D:`
5. 错误手册——上一步前半部分成功，万事大吉开始学习；前半部分失败而后者成功那就是vsc没有切换环境，重新切换环境ctrl+p,>python:select...选择新建的环境解释器即可
## 错误手册
1. prompt中检查torch安装成功并可以输出版本，但是在vsc中导入失败
* 错误代码
import torch
ModuleNotFoundError: No module named 'torch'
* 卸载环境后重装，在终端仍旧显示找不到模块，但是coderunner可以运行
2. 打开根目录进行运行失败
*仍旧提示找不到模块，在prompt进行测试成功输出了torch的版本
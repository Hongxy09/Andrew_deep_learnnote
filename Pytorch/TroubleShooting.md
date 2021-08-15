1. 问题：prompt中检查torch安装成功并可以输出版本，但是在vsc中导入失败
* 错误代码
```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
If using 'conda activate' from a batch script, change your
invocation to 'CALL conda.bat activate'.
To initialize your shell, run
   $ conda init <SHELL_NAME>
   Currently supported shells are:
  - bash
  - cmd.exe
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell
ModuleNotFoundError: No module named 'torch'
from numpy.core._multiarray_umath import
ImportError: DLL load failed: 找不到指定的模块。
```
* 问题诊断
仍旧提示找不到模块，在prompt进行测试成功输出了torch的版本；直接用prompt运行脚本文件成功。错误提示的意思是说，vsc的终端不能正常使用```conda activate```指令，应当在初次使用虚拟环境调用时进行终端初始化。
* 解决方法
vsc终端是powershell，因此在powershell中初始化后，测试是否能够运行。问题解决。
2. 问题：导入数据集解压的时候多进程引发错误
* 错误代码
```
An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your child processes and you have forgotten to use the proper idiom in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
```
* 解决方法
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)的子进程数目调整为0即可
3. 问题：anaconda图形界面打开卡住
anaconda3\Lib\site-packages\anaconda_navigator\api\conda_api.py 行1364 把 data = yaml.load(f) 改为 data = yaml.safeload(f)
4. 问题：github同步失败
* 错误代码
a.RPC failed; curl 18 transfer closed with outstanding read data remaining
send-pack: unexpected disconnect while reading sideband packet
fatal: the remote end hung up unexpectedly
b.fatal: unable to access 'https://github.com/Hongxy09/Deeplearn_note.git/': OpenSSL SSL_read: Connection was reset, errno 10054
* 问题诊断：提示网络断开，远程挂起
* 解决方法：重新提交、清除DNS```缓存ipconfig /flushdns```
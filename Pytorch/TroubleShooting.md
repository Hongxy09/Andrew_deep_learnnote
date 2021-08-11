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
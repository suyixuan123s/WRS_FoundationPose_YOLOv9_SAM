'''


取消已设置的环境变量
如果你已经通过 export 设置了某些环境变量，可以使用 unset 命令来取消它们。具体操作如下：

取消 CMAKE_PREFIX_PATH 环境变量：

bash
复制代码
unset CMAKE_PREFIX_PATH
取消 BOOST_ROOT 环境变量：

bash
复制代码
unset BOOST_ROOT
取消 EIGEN3_INCLUDE_DIR 环境变量：

bash
复制代码
unset EIGEN3_INCLUDE_DIR
取消 PYBIND11_INCLUDE_DIR 环境变量：

bash
复制代码
unset PYBIND11_INCLUDE_DIR
检查取消是否成功
你可以通过 echo 命令检查这些环境变量是否已经被取消：

bash
复制代码
echo $CMAKE_PREFIX_PATH
echo $BOOST_ROOT
echo $EIGEN3_INCLUDE_DIR
echo $PYBIND11_INCLUDE_DIR
如果这些变量已经被取消，它们的输出应该为空。


'''
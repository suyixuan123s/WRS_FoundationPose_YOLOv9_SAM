'''
export PATH=$PATH:/usr/local/cuda-11.8/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64  分析我使用的是 sudo  nano ~/.bashrc 怎样保存退出
在使用 sudo nano ~/.bashrc 编辑文件后，保存并退出的方法如下：

保存文件：

按下 Ctrl + O（即字母 O），这是保存文件的快捷键。
在屏幕底部，会提示你文件名，默认情况下是 ~/.bashrc。你可以直接按回车键来确认保存。退出编辑器：

按下 Ctrl + X 来退出 nano 编辑器。
完成这些步骤后，.bashrc 文件就会被保存并且退出。为了使更改生效，你可以执行以下命令重新加载配置文件：


source ~/.bashrc

'''

'''
# create conda environment
conda create -n foundationpose python=3.9

# activate conda environment
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment

conda install conda-forge::eigen=3.4.0

如果你不确定安装路径，可以使用 find 命令在 Conda 环境目录中查找 eigen 相关的文件：

find $CONDA_PREFIX -name eigen


(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~$ find $CONDA_PREFIX -name eigen
(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~$ ls $CONDA_PREFIX/include/eigen3
Eigen  signature_of_eigen3_matrix_library  unsupported
(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~$ conda list eigen
# packages in environment at /home/suyixuan_sam/anaconda3/envs/foundationpose:
#
# Name                    Version                   Build  Channel
eigen                     3.4.0                h00ab1b0_0    conda-forge
(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~$ find $CONDA_PREFIX -name eigen
(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~$

************************************************************************************************************************

(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~$ ls $CONDA_PREFIX/include/eigen3/Eigen
Cholesky        Eigen        IterativeLinearSolvers  MetisSupport     QR               SparseCore   src        SuperLUSupport
CholmodSupport  Eigenvalues  Jacobi                  OrderingMethods  QtAlignedMalloc  SparseLU     StdDeque   SVD
Core            Geometry     KLUSupport              PardisoSupport   Sparse           SparseQR     StdList    UmfPackSupport
Dense           Householder  LU                      PaStiXSupport    SparseCholesky   SPQRSupport  StdVector
(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~$  分析我的路径怎样设置

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX/include/eigen3"



export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX/include/eigen3/Eigen
--------------------------------------------------------------------------------------------------------------------------
# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

'''


'''
(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~/AI/Pose_Estimation/FoundationPose$ ls /usr/local/boost
ls: 无法访问 '/usr/local/boost': 没有那个文件或目录
(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~/AI/Pose_Estimation/FoundationPose$ find $CONDA_PREFIX -name boost
/home/suyixuan_sam/anaconda3/envs/foundationpose/lib/python3.9/site-packages/cmeel.prefix/include/boost
/home/suyixuan_sam/anaconda3/envs/foundationpose/lib/python3.9/site-packages/cmeel.prefix/include/boost/chrono/typeof/boost
/home/suyixuan_sam/anaconda3/envs/foundationpose/lib/python3.9/site-packages/cmeel.prefix/include/boost/hana/ext/boost
/home/suyixuan_sam/anaconda3/envs/foundationpose/lib/python3.9/site-packages/cmeel.prefix/lib/cmake/eigenpy/boost
(foundationpose) suyixuan_sam@suyixuan_sam-ASUS-TUF-Gaming-F16-FX607JVR-FX607JVR:~/AI/Pose_Estimation/FoundationPose$ sudo apt-get remove --purge libboost-all-dev


/home/suyixuan_sam/anaconda3/envs/foundationpose1/lib/python3.9/site-packages/pybind11/share/cmake/pybind11:/home/suyixuan_sam/anaconda3/envs/foundationpose1:/home/suyixuan_sam/anaconda3/envs/foundationpose1/lib:/home/suyixuan_sam/anaconda3/envs/foundationpose1/include:/home/suyixuan_sam/anaconda3/envs/foundationpose1/include/boost:/home/suyixuan_sam/anaconda3/envs/foundationpose1/include/eigen3/Eigen

/home/suyixuan_sam/anaconda3/envs/foundationpose1/lib/python3.9/site-packages/pybind11/share/cmake/pybind11
/home/suyixuan_sam/anaconda3/envs/foundationpose1
/home/suyixuan_sam/anaconda3/envs/foundationpose1/lib
/home/suyixuan_sam/anaconda3/envs/foundationpose1/include
/home/suyixuan_sam/anaconda3/envs/foundationpose1/include/boost
/home/suyixuan_sam/anaconda3/envs/foundationpose1/include/eigen3

'''
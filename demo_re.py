#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ABB_wrs_hu 
@File    ：demo_re.py
@IDE     ：PyCharm 
@Author  ：suyixuan
@Date    ：2025-05-13 21:28:16

WRS_FoundationPose_YOLOv9_SAM
) suyixuan@suyixuan-LEGION-REN9000K-34IRZ:~/AI/pytracik$ conda list
# packages in environment at /home/suyixuan/anaconda3/envs/WRS_FoundationPose_YOLOv9_SAM
:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main
_openmp_mutex             5.1                       1_gnu
addict                    2.4.0                    pypi_0    pypi
aiohappyeyeballs          2.6.1                    pypi_0    pypi
aiohttp                   3.11.18                  pypi_0    pypi
aiosignal                 1.3.2                    pypi_0    pypi
albumentations            1.4.2                    pypi_0    pypi
antlr4-python3-runtime    4.9.3                    pypi_0    pypi
anyio                     4.9.0                    pypi_0    pypi
appdirs                   1.4.4                    pypi_0    pypi
argon2-cffi               23.1.0                   pypi_0    pypi
argon2-cffi-bindings      21.2.0                   pypi_0    pypi
arrow                     1.3.0                    pypi_0    pypi
asttokens                 3.0.0                    pypi_0    pypi
async-lru                 2.0.5                    pypi_0    pypi
async-timeout             5.0.1                    pypi_0    pypi
attrs                     25.3.0                   pypi_0    pypi
babel                     2.17.0                   pypi_0    pypi
beautifulsoup4            4.13.4                   pypi_0    pypi
bleach                    6.2.0                    pypi_0    pypi
blinker                   1.9.0                    pypi_0    pypi
bokeh                     3.4.0                    pypi_0    pypi
braceexpand               0.1.7                    pypi_0    pypi
brotli                    1.1.0                    pypi_0    pypi
ca-certificates           2025.4.26            hbd8a1cb_0    conda-forge
certifi                   2025.4.26                pypi_0    pypi
cffi                      1.17.1                   pypi_0    pypi
charset-normalizer        3.4.2                    pypi_0    pypi
click                     8.1.8                    pypi_0    pypi
cmake                     4.0.2                    pypi_0    pypi
cmeel                     0.57.3                   pypi_0    pypi
cmeel-assimp              5.4.3.1                  pypi_0    pypi
cmeel-boost               1.83.0                   pypi_0    pypi
cmeel-console-bridge      1.0.2.3                  pypi_0    pypi
cmeel-octomap             1.10.0                   pypi_0    pypi
cmeel-qhull               8.0.2.1                  pypi_0    pypi
cmeel-tinyxml             2.6.2.3                  pypi_0    pypi
cmeel-urdfdom             3.1.1.1                  pypi_0    pypi
cmeel-zlib                1.3.1                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
coloredlogs               15.0.1                   pypi_0    pypi
comm                      0.2.2                    pypi_0    pypi
common                    0.0.0                     dev_0    <develop>
configargparse            1.7                      pypi_0    pypi
console_bridge            1.0.2                h924138e_1    conda-forge
contourpy                 1.3.0                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
dash                      3.0.4                    pypi_0    pypi
dataclasses-json          0.6.7                    pypi_0    pypi
debugpy                   1.8.14                   pypi_0    pypi
decorator                 5.2.1                    pypi_0    pypi
defusedxml                0.7.1                    pypi_0    pypi
deprecated                1.2.18                   pypi_0    pypi
docker-pycreds            0.4.0                    pypi_0    pypi
eigen                     3.4.0                h00ab1b0_0    conda-forge
eigenpy                   3.5.1                    pypi_0    pypi
einops                    0.7.0                    pypi_0    pypi
entrypoints               0.4                      pypi_0    pypi
et-xmlfile                2.0.0                    pypi_0    pypi
exceptiongroup            1.3.0                    pypi_0    pypi
executing                 2.2.0                    pypi_0    pypi
fastjsonschema            2.21.1                   pypi_0    pypi
ffmpeg-python             0.2.0                    pypi_0    pypi
filelock                  3.18.0                   pypi_0    pypi
flask                     3.0.3                    pypi_0    pypi
flatbuffers               25.2.10                  pypi_0    pypi
fonttools                 4.58.0                   pypi_0    pypi
fqdn                      1.5.1                    pypi_0    pypi
freetype-py               2.5.1                    pypi_0    pypi
frozenlist                1.6.0                    pypi_0    pypi
fsspec                    2025.3.2                 pypi_0    pypi
future                    1.0.0                    pypi_0    pypi
fvcore                    0.1.5.post20221221          pypi_0    pypi
g4f                       0.2.7.1                  pypi_0    pypi
gitdb                     4.0.12                   pypi_0    pypi
gitpython                 3.1.44                   pypi_0    pypi
gputil                    1.4.0                    pypi_0    pypi
h11                       0.16.0                   pypi_0    pypi
h5py                      3.10.0                   pypi_0    pypi
hf-xet                    1.1.0                    pypi_0    pypi
hpp-fcl                   2.4.4                    pypi_0    pypi
httpcore                  1.0.9                    pypi_0    pypi
httpx                     0.28.1                   pypi_0    pypi
huggingface-hub           0.31.1                   pypi_0    pypi
humanfriendly             10.0                     pypi_0    pypi
idna                      3.10                     pypi_0    pypi
imageio                   2.34.0                   pypi_0    pypi
imgaug                    0.4.0                    pypi_0    pypi
importlib-metadata        8.7.0                    pypi_0    pypi
importlib-resources       6.5.2                    pypi_0    pypi
iopath                    0.1.10                   pypi_0    pypi
ipycanvas                 0.13.3                   pypi_0    pypi
ipyevents                 2.0.2                    pypi_0    pypi
ipykernel                 6.29.5                   pypi_0    pypi
ipython                   8.18.1                   pypi_0    pypi
ipywidgets                8.1.2                    pypi_0    pypi
iso8601                   2.1.0                    pypi_0    pypi
isoduration               20.11.0                  pypi_0    pypi
itsdangerous              2.2.0                    pypi_0    pypi
jedi                      0.19.2                   pypi_0    pypi
jinja2                    3.1.6                    pypi_0    pypi
joblib                    1.3.2                    pypi_0    pypi
json5                     0.12.0                   pypi_0    pypi
jsonpatch                 1.33                     pypi_0    pypi
jsonpointer               3.0.0                    pypi_0    pypi
jsonschema                4.23.0                   pypi_0    pypi
jsonschema-specifications 2025.4.1                 pypi_0    pypi
jupyter-client            7.4.9                    pypi_0    pypi
jupyter-core              5.7.2                    pypi_0    pypi
jupyter-events            0.12.0                   pypi_0    pypi
jupyter-lsp               2.2.5                    pypi_0    pypi
jupyter-server            2.15.0                   pypi_0    pypi
jupyter-server-terminals  0.5.3                    pypi_0    pypi
jupyterlab                4.1.5                    pypi_0    pypi
jupyterlab-pygments       0.3.0                    pypi_0    pypi
jupyterlab-server         2.27.3                   pypi_0    pypi
jupyterlab-widgets        3.0.15                   pypi_0    pypi
kaolin                    0.15.0                   pypi_0    pypi
keyboard                  0.13.5                   pypi_0    pypi
kiwisolver                1.4.7                    pypi_0    pypi
kornia                    0.7.2                    pypi_0    pypi
kornia-rs                 0.1.9                    pypi_0    pypi
lazy-loader               0.4                      pypi_0    pypi
ld_impl_linux-64          2.40                 h12ee557_0
libffi                    3.4.4                h6a678d5_1
libgcc                    15.1.0               h767d61c_2    conda-forge
libgcc-ng                 15.1.0               h69a702a_2    conda-forge
libgomp                   15.1.0               h767d61c_2    conda-forge
libstdcxx                 15.1.0               h8f9b012_2    conda-forge
libstdcxx-ng              15.1.0               h4852527_2    conda-forge
lit                       18.1.8                   pypi_0    pypi
llvmlite                  0.42.0                   pypi_0    pypi
loguru                    0.7.3                    pypi_0    pypi
lxml                      5.2.2                    pypi_0    pypi
markupsafe                3.0.2                    pypi_0    pypi
marshmallow               3.26.1                   pypi_0    pypi
matplotlib                3.9.4                    pypi_0    pypi
matplotlib-inline         0.1.7                    pypi_0    pypi
meshcat                   0.3.2                    pypi_0    pypi
mistune                   3.1.3                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
multidict                 6.4.3                    pypi_0    pypi
mypy-extensions           1.1.0                    pypi_0    pypi
narwhals                  1.39.0                   pypi_0    pypi
nbclient                  0.10.2                   pypi_0    pypi
nbconvert                 7.16.6                   pypi_0    pypi
nbformat                  5.10.4                   pypi_0    pypi
ncurses                   6.4                  h6a678d5_0
nest-asyncio              1.6.0                    pypi_0    pypi
networkx                  3.2.1                    pypi_0    pypi
ninja                     1.11.1.3                 pypi_0    pypi
nlopt                     2.7.1                h6a678d5_0
nodejs                    0.1.1                    pypi_0    pypi
notebook-shim             0.2.4                    pypi_0    pypi
numba                     0.59.1                   pypi_0    pypi
numpy                     1.26.4                   pypi_0    pypi
nvdiffrast                0.3.3                    pypi_0    pypi
nvidia-cublas-cu12        12.6.4.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.6.80                  pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.6.77                  pypi_0    pypi
nvidia-cuda-runtime-cu12  12.6.77                  pypi_0    pypi
nvidia-cudnn-cu12         9.5.1.17                 pypi_0    pypi
nvidia-cufft-cu12         11.3.0.4                 pypi_0    pypi
nvidia-cufile-cu12        1.11.1.6                 pypi_0    pypi
nvidia-curand-cu12        10.3.7.77                pypi_0    pypi
nvidia-cusolver-cu12      11.7.1.2                 pypi_0    pypi
nvidia-cusparse-cu12      12.5.4.2                 pypi_0    pypi
nvidia-cusparselt-cu12    0.6.3                    pypi_0    pypi
nvidia-nccl-cu12          2.26.2                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.6.85                  pypi_0    pypi
nvidia-nvtx-cu12          12.6.77                  pypi_0    pypi
objaverse                 0.1.7                    pypi_0    pypi
omegaconf                 2.3.0                    pypi_0    pypi
onnxruntime               1.18.1                   pypi_0    pypi
open3d                    0.18.0                   pypi_0    pypi
opencv-contrib-python     4.9.0.80                 pypi_0    pypi
opencv-python             4.9.0.80                 pypi_0    pypi
opencv-python-headless    4.11.0.86                pypi_0    pypi
openpyxl                  3.1.2                    pypi_0    pypi
openssl                   3.5.0                h7b32b05_1    conda-forge
optional-django           0.1.0                    pypi_0    pypi
overrides                 7.7.0                    pypi_0    pypi
packaging                 25.0                     pypi_0    pypi
panda3d                   1.10.14                  pypi_0    pypi
pandas                    2.2.3                    pypi_0    pypi
pandocfilters             1.5.1                    pypi_0    pypi
parso                     0.8.4                    pypi_0    pypi
pexpect                   4.9.0                    pypi_0    pypi
pillow                    11.2.1                   pypi_0    pypi
pin                       2.7.0                    pypi_0    pypi
pip                       25.1               pyhc872135_2
platformdirs              4.3.8                    pypi_0    pypi
plotly                    5.20.0                   pypi_0    pypi
portalocker               3.1.1                    pypi_0    pypi
prometheus-client         0.21.1                   pypi_0    pypi
prompt-toolkit            3.0.51                   pypi_0    pypi
propcache                 0.3.1                    pypi_0    pypi
protobuf                  4.25.7                   pypi_0    pypi
psutil                    6.0.0                    pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pure-eval                 0.2.3                    pypi_0    pypi
py-cpuinfo                9.0.0                    pypi_0    pypi
py-spy                    0.3.14                   pypi_0    pypi
pyapriltags               3.3.0.3                  pypi_0    pypi
pyarrow                   20.0.0                   pypi_0    pypi
pybind11                  2.12.0                   pypi_0    pypi
pybullet                  3.2.6                    pypi_0    pypi
pycocotools               2.0.7                    pypi_0    pypi
pycparser                 2.22                     pypi_0    pypi
pycryptodome              3.22.0                   pypi_0    pypi
pyglet                    1.5.28                   pypi_0    pypi
pygltflib                 1.16.4                   pypi_0    pypi
pygments                  2.19.1                   pypi_0    pypi
pyngrok                   7.2.8                    pypi_0    pypi
pyopengl                  3.1.0                    pypi_0    pypi
pyopengl-accelerate       3.1.9                    pypi_0    pypi
pyparsing                 3.2.3                    pypi_0    pypi
pypng                     0.20220715.0             pypi_0    pypi
pyquaternion              0.9.9                    pypi_0    pypi
pyreadline                2.1                      pypi_0    pypi
pyreadline3               3.5.4                    pypi_0    pypi
pyrealsense2              2.55.1.6486              pypi_0    pypi
pyrender                  0.1.45                   pypi_0    pypi
pysdf                     0.1.9                    pypi_0    pypi
pyserial                  3.5                      pypi_0    pypi
python                    3.9.21               he870216_1
python-dateutil           2.9.0.post0              pypi_0    pypi
python-json-logger        3.3.0                    pypi_0    pypi
pytorch3d                 0.7.3                    pypi_0    pypi
pytracik                  0.0.1                    pypi_0    pypi
pytz                      2025.2                   pypi_0    pypi
pyyaml                    6.0.1                    pypi_0    pypi
pyzmq                     24.0.1                   pypi_0    pypi
readline                  8.2                  h5eee18b_0
referencing               0.36.2                   pypi_0    pypi
requests                  2.32.3                   pypi_0    pypi
retrying                  1.3.4                    pypi_0    pypi
rfc3339-validator         0.1.4                    pypi_0    pypi
rfc3986-validator         0.1.1                    pypi_0    pypi
roma                      1.4.4                    pypi_0    pypi
rpds-py                   0.24.0                   pypi_0    pypi
rtree                     1.2.0                    pypi_0    pypi
ruamel-yaml               0.18.6                   pypi_0    pypi
ruamel-yaml-clib          0.2.12                   pypi_0    pypi
safetensors               0.5.3                    pypi_0    pypi
scikit-image              0.22.0                   pypi_0    pypi
scikit-learn              1.4.1.post1              pypi_0    pypi
scipy                     1.12.0                   pypi_0    pypi
seaborn                   0.13.2                   pypi_0    pypi
send2trash                1.8.3                    pypi_0    pypi
sentry-sdk                2.28.0                   pypi_0    pypi
serial                    0.0.97                   pypi_0    pypi
setproctitle              1.3.6                    pypi_0    pypi
setuptools                69.5.1                   pypi_0    pypi
shapely                   2.0.7                    pypi_0    pypi
simplejson                3.19.2                   pypi_0    pypi
six                       1.17.0                   pypi_0    pypi
smmap                     5.0.2                    pypi_0    pypi
sniffio                   1.3.1                    pypi_0    pypi
soupsieve                 2.7                      pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0
stack-data                0.6.3                    pypi_0    pypi
sympy                     1.14.0                   pypi_0    pypi
tabulate                  0.9.0                    pypi_0    pypi
tenacity                  9.1.2                    pypi_0    pypi
termcolor                 3.1.0                    pypi_0    pypi
terminado                 0.18.1                   pypi_0    pypi
thop                      0.1.1-2209072238          pypi_0    pypi
threadpoolctl             3.6.0                    pypi_0    pypi
tifffile                  2024.8.30                pypi_0    pypi
timm                      0.9.16                   pypi_0    pypi
tinycss2                  1.4.0                    pypi_0    pypi
tinyxml2                  11.0.0               h3f2d84a_0    conda-forge
tk                        8.6.14               h39e8969_0
tomli                     2.2.1                    pypi_0    pypi
torch                     2.0.0+cu118              pypi_0    pypi
torchaudio                2.0.1+cu118              pypi_0    pypi
torchnet                  0.0.4                    pypi_0    pypi
torchvision               0.15.1+cu118             pypi_0    pypi
tornado                   6.4.2                    pypi_0    pypi
tqdm                      4.67.1                   pypi_0    pypi
traitlets                 5.14.3                   pypi_0    pypi
transformations           2024.6.1                 pypi_0    pypi
trimesh                   4.2.2                    pypi_0    pypi
triton                    2.0.0                    pypi_0    pypi
types-python-dateutil     2.9.0.20241206           pypi_0    pypi
typing-extensions         4.13.2                   pypi_0    pypi
typing-inspect            0.9.0                    pypi_0    pypi
tzdata                    2025.2                   pypi_0    pypi
u-msgpack-python          2.8.0                    pypi_0    pypi
ultralytics               8.0.120                  pypi_0    pypi
ultralytics-thop          2.0.14                   pypi_0    pypi
urdfdom                   4.0.1                hae71d53_3    conda-forge
urdfdom_headers           1.1.2                h84d6215_0    conda-forge
uri-template              1.3.0                    pypi_0    pypi
urllib3                   2.4.0                    pypi_0    pypi
usd-core                  23.5                     pypi_0    pypi
videoio                   0.2.8                    pypi_0    pypi
visdom                    0.2.4                    pypi_0    pypi
wandb                     0.16.5                   pypi_0    pypi
warp-lang                 1.0.2                    pypi_0    pypi
wcwidth                   0.2.13                   pypi_0    pypi
webcolors                 24.11.1                  pypi_0    pypi
webdataset                0.2.86                   pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.8.0                    pypi_0    pypi
werkzeug                  3.0.6                    pypi_0    pypi
wheel                     0.45.1           py39h06a4308_0
widgetsnbextension        4.0.14                   pypi_0    pypi
wrapt                     1.17.2                   pypi_0    pypi
xatlas                    0.0.9                    pypi_0    pypi
xlsxwriter                3.2.0                    pypi_0    pypi
xyzservices               2025.4.0                 pypi_0    pypi
xz                        5.6.4                h5eee18b_1
yacs                      0.1.8                    pypi_0    pypi
yarl                      1.20.0                   pypi_0    pypi
zipp                      3.21.0                   pypi_0    pypi
zlib                      1.2.13               h5eee18b_1
(WRS_FoundationPose_YOLOv9_SAM
) suyixuan@suyixuan-LEGION-REN9000K-34IRZ:~/AI/pytracik$



'''

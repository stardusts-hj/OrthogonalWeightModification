git clone 代码到 gwork 里面

启动docker

选择的镜像是yujr-reckit-tf114

```
startdocker -u "-it" -c /bin/bash bit:5000/yujr-reckit-tf114

startdocker -u "-it" -c /bin/bash bit:5000/cuihao_titanxp_py3_deeplearning
```







然后在容器中，运行代码

```
python expt_owm.py -owm task
```
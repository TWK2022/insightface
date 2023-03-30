## 快速建立人脸特征数据库并进行人脸检测和身份识别
>基于insightface官方项目改编:https://github.com/deepinsight/insightface  
>首次运行代码时会自动下载模型文件到用户下的.insightface文件夹中  
### 1，database_prepare.py
>将人脸数据库图片放入文件夹image_database中，最好是证件照  
>运行database_prepare.py即可生成人脸特征数据库feature_database.csv
### 2，predict.py
>模式1: args中设置camera=False  
>将待预测图片放入文件夹image_predict中，运行predict.py  
>模式2: args中设置camera=True  
>运行predict.py，将使用电脑视像头，实时预测视像头中截取的画面并显示
### 其他
>github链接:https://github.com/TWK2022/insightface  
>学习笔记:https://github.com/TWK2022/notebook  
>邮箱:1024565378@qq.com
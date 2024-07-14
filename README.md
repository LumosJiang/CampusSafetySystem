# CampusSafetySystem
北京交通大学软件学院2024年暑期实训-一个响亮的名字


Dlib_face_recognition_from_camara中的data文件夹下还有一个权重，需要前往百度网盘自行下载。
链接：https://pan.baidu.com/s/17qjW0SLSAQZsb_ooZ9jcrw 
提取码：mlvx


yolov7_reid-master\person_search\weights下的为reid权重，
链接：https://pan.baidu.com/s/1EbhFiLe_HDuj9CtI1rLKOA 
提取码：mlvx

yolov7_reid-master\person_search下有一个yolov7权重
链接：https://pan.baidu.com/s/1MdmzrD5OH4m7vuRENOFTpQ 
提取码：mlvx

yolov5-5.0过大，放到百度网盘，自行下载
链接：https://pan.baidu.com/s/13SOTzRVWqHkwFNeD4UMbuA 
提取码：mlvx

前端页面Frontend放到百度网盘。自行下载
链接：https://pan.baidu.com/s/15Y3mPqWbHARA4oJxZDLbeA 
提取码：mlvx

数据大屏的WEB-INF放到百度网盘。自行下载。
链接：https://pan.baidu.com/s/16xn7rblEuW5myiBY1qgPqg 
提取码：mlvx


> # Dlib_face_recognition_from_camera-master为人脸识别模块代码

其中，

register_finally.py为人脸录入代码

face_finally.py为人脸识别代码

face_finally_life.py为人脸识别+活体检测代码

white_black_list.py为黑白名单代码

想要在前端页面进行人脸识别功能演示，首先要在后端运行该四个代码

其中，face_finally.py和face_finally_life.py的如下配置需要改成自己的mysql配置。user为用户名，password为密码，database为数据库名称

```
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'mydatabase'
}
```

其中，

face_finally.py的481行需要将rtmp地址改为自己推流的地址，这里默认是live1，face_finally_life.py的491行需要将rtmp地址改为自己推流的地址，这里默认是live1,

register_finally.py的120行需要将rtmp地址改为自己推流的地址，这里默认是live1



face_finally.py中的第94、119、152、169、186行需要改成vue代码所在的对应位置。

face_finally_life.py中的98,、123、156、173、189同理，也需要改为对应的位置

> SQL语句脚本

```mysql
create database mydatabase;
use mydatabase;
create table user(
    user_id int primary key ,
    name varchar(20),
    password varchar(20),
    email varchar(20),
    authority int
);
CREATE TABLE Video (
   video_id INT PRIMARY KEY AUTO_INCREMENT,
   path VARCHAR(100),
   date DATETIME,
   name VARCHAR(40),
   type VARCHAR(8)
);

create table 教学楼(
    楼名称 varchar(20) primary key
);
insert into 教学楼 values
('逸夫教学楼 & 研究生院'),
('北京交通大学综合体育馆'),
('思源西楼'),
('思源东楼 '),
('思源楼 '),
('18号学生公寓'),
('图书馆'),
('16号学生公寓'),
('第四教学楼'),
('第五教学楼'),
('12号学生公寓'),
('19号学生公寓'),
('2号学生公寓'),
('第七教学楼'),
('第九教学楼'),
('20号宿舍楼'),
('体育馆'),
('22号学生公寓'),
('交大知行大厦'),
('嘉园C座'),
('留园公寓'),
('机械楼'),
('电气工程楼'),
('嘉园B座'),
('嘉园A座'),
('天佑会堂'),
('运输设备教学馆');

create table 安防设备详情(
    设备编号 varchar(5) primary key ,
    设备名称 varchar(10),
    设备状态 varchar(2)
);

insert into 安防设备详情 values
('CB001','第八教学楼','正常'),
('CB002','土木工程楼','异常'),
('CB003','传习路','正常'),
('CB004','银杏大道','异常'),
('CB005','西门','正常');

create table 安防设备监测(
    分类 varchar(5) primary key ,
    数量 int
);
insert into 安防设备监测 values
    ('正常',754),
    ('出现异常',24);

create table 异常事件类型统计(
    事件类型 varchar(10) primary key ,
    数量 int
);
insert into 异常事件类型统计 values
('跌倒',8),
('着火',2),
('黑名单人员',6),
('非活体入侵',5);

create table 异常事件列表(
    时间 datetime primary key ,
    地点 varchar(20) ,
    事件 varchar(10) ,
    address varchar(100)
);

insert into 异常事件列表 values
('2024-07-08 08:12:43','第八教学楼','跌倒','../../public/pictures/fobbiden/output_0.jpg'),
('2024-07-08 12:12:00','西门','跌倒','../../public/pictures/fobbiden/output_1.jpg'),
('2024-07-08 12:20:00','土木工程楼','着火','../../public/pictures/fobbiden/output_2.jpg'),
('2024-07-08 03:52:22','西门','非活体入侵','../../public/pictures/fobbiden/output_3.jpg'),
('2024-07-08 19:19:43','传习路','跌倒','../../public/pictures/fobbiden/output_4.jpg'),
('2024-07-11 08:12:43','土木工程楼','跌倒','../../public/pictures/fobbiden/output_5.jpg'),
('2024-07-12 09:12:54','银杏大道','跌倒','../../public/pictures/fobbiden/output_6.jpg');
create table 摄像头位置(
    摄像头 varchar(10) primary key ,
    经度 varchar(15),
    纬度 varchar(15),
    状态 varchar(2)
);
insert into 摄像头位置 values
('传习路','116.334193','39.94114','正常'),
('土木工程楼','116.339496', '39.949114','异常'),
('第八教学楼','116.335853','39.951354','正常'),
('西门','116.331165','39.950499','正常'),
('银杏大道','116.336269','39.950075','异常');
create table 教学楼1(
    楼名称 varchar(20) primary key
);
insert into 教学楼1 values
('逸夫教学楼 & 研究生院'),
('16号学生公寓'),
('图书馆');

```

> # fastApiyolo为异常行为检测代码

其中main.py为运行界面，想要在前端实现异常行为检测，首先需要在后端运行该代码。

main.py的69、104、122、147行代码需要做对应的更改

> #  yolov5-5.0为目标追踪画框代码

其中，detect.py为运行代码，想要在前端实现爱你目标追踪区域框定，首先要在后端运行该代码

第29行的

```
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'mydatabase'
}
```

改为自己的配置。

54、78、93、118、134、183、251、277都需要进行对应的更改。

> # yolov7_reid-master为reid代码

其中，person_seach文件夹下的serch4.py为运行代码，想要在前端实现目标reid，首先要在后端运行该代码。

将source.txt中的流改为自己的流，然后运行即可。

其中，backreid.py为只在后端能运行的代码，即可以在后端显示reid结果。

> # Frontend.zip 为前端页面代码

首先需要将文件夹进行解压缩，安装node.js,然后用vscode打开，在终端输入

```
npm run dev
```

即可运行该前端代码

> # demo111为javaSpringboot后端代码

首先需要将application.yml中的

```
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/mydatabase
    username: root
    password: 123456

```

改为自己对应的jdbc与数据库的相关参数

然后直接运行即可。

> 数据大屏

首先下载帆软app，然后将webapps/webroot目录下的WEB_INF替换为压缩包WEB_INF.zip解压缩之后的内容。

打开帆软之后，将smartCampus.fvs添加到工作目录，点击右上角的预览。

输入账号名和密码

```
admin
123456
```

选择保持登录状态。然后即可在前端页面进行数据大屏的展示。

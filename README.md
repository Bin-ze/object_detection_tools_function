# object_detection_tools_function
描述：一些目标检测任务需要用到的数据转换以及常见数据处理函数
##  statistics_class.py
描述：用于对 xml(voc)，json(coco)，txt(yolo) 格式的数据集的每个类别的数量进行统计，并输出json
## voc_split_trainval.py
描述：用于随机划分voc格式的目标检测数据集的train set和val set
## voc2coco.py
描述：实现了voc数据集的xml格式到coco数据集的json格式的数据转换
## mmdet_infer.py
描述：借助mmdeploy实现模型部署之后的模型推理流程，实现了包括推理，画框，可视化的流程
## compute_pred_acc.py
描述：计算yolo标注的图像漏检以及错检文件，输出准确率指标，添加了格式化输出，以及检测结果可视化，错误定位，过滤无效类别等功能。
## coco2txt.py
描述：coco数据集转yolo格式
## yolo2coco.py
描述：yolo标注转coco的json标注
## handle_yolo_txt_path.py
描述：用于生成yolo的txt文件
## create_dataset.py
描述：用于将目标检测数据集转换为分类数据集，训练分类模型
## single_to_all.py
描述：用于实现单天菜品到整个菜品数据集的映射
## val_14class_to_120class.py
描述：用于实现单天14类标注到120类整个数据集标注的标签转换
## voc2yolo.py
描述：用于voc的xml标注到yolo的txt标注的转换
## json_to_yolopose.py
描述：用于将json标注的关键点转化为yolopose格式

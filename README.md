# object_detection_tools_function
  一些目标检测任务需要用到的数据转换以及常见数据处理函数
##  statistics_class.py
用于对 xml(voc)，json(coco)，txt(yolo) 格式的数据集的每个类别的数量进行统计，并输出json
## voc_split_trainval.py
用于随机划分voc格式的目标检测数据集的train set和val set
## voc2coco.py
实现了voc数据集的xml格式到coco数据集的json格式的数据转换
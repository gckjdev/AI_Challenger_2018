#!/user/bin/env python
# -*- coding:utf-8 -*-

import os
model_save_path = os.path.abspath('.') + "/data/"
train_data_path = os.path.abspath('.') +  "/training/sentiment_analysis_trainingset.csv"  # "训练集文件存放路径"
validate_data_path = os.path.abspath('.') + "/validate/sentiment_analysis_validationset.csv" # "验证集文件存放路径"
test_data_path = os.path.abspath('.') + "/testa/sentiment_analysis_testa.csv" # "测试集文件存放路径"
test_data_predict_out_path = os.path.abspath('.') + "/output/out.csv" # "测试集预测结果文件存放路径"

user_dict_path = os.path.abspath('.') + "/userdict.txt"


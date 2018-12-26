# -*- coding: utf-8 -*-

import os
import configparser



# sections = ["离散化" , "标准化"]
# options = [
#     {"culumns":"年龄" , "aaa":"10"} ,
#     {"columns":"上船地点" , "giy":"够用"}
# ]
def creat_ini(filePath , sections , options , encoding="utf-8"):
    fileName = filePath.split("\\")[-1]
    driectory = filePath.strip(fileName)[0:-1]
    is_exists = os.path.exists(driectory)  # 判断一个目录是否存在
    if is_exists is False:
        os.makedirs(driectory)  # 创建目录
    is_exists = os.path.exists(filePath)
    if is_exists is True:
        red_color = "\33[1;31m"
        default_color = "\33[0m"
        warning = red_color + fileName + " 已经存在，创建失败。如果想要修改，请调用 update_ini() 函数" + default_color
        return print(warning)
    config = configparser.ConfigParser()
    for i in range(len(sections)):
        config.add_section(sections[i])
        key_list = []
        values_list = []
        for key in options[i].keys():
            key_list.append(key)
        for value in options[i].values():
            values_list.append(value)
        for j in range(len(key_list)):
            config.set(sections[i] , key_list[j] , values_list[j])
    file = open(filePath , "w" , encoding=encoding)
    config.write(file)
    file.close()





def read_ini(filePath , encoding="utf-8"):
    config = configparser.ConfigParser()
    file = open(filePath, "r", encoding=encoding)
    config.read_file(file)
    file.close()
    sections = config.sections()
    options = []
    for i in range(len(sections)):
        option_keys = config.options(sections[i])
        option = {}
        for j in range(len(option_keys)):
            option_value = config.get(sections[i] , option_keys[j])
            option[option_keys[j]] = option_value
        options.append(option)
    return sections , options





def update_ini(filePath , sections , options , encoding="utf-8"):
    config = configparser.ConfigParser()
    file = open(filePath , "r" , encoding=encoding)
    config.read_file(file)
    file.close()
    for i in range(len(sections)):
        is_exists_section = config.has_section(sections[i]) #是否存在 sections[i]
        if is_exists_section is False:
            config.add_section(sections[i]) #如果不存在就创建 sections[i]
        key_list = []
        values_list = []
        for key in options[i].keys():
            key_list.append(key)
        for value in options[i].values():
            values_list.append(value)
        for j in range(len(key_list)):
            config.set(sections[i] , key_list[j] , values_list[j])
    file = open(filePath , "w" , encoding=encoding)
    config.write(file)
    file.close()


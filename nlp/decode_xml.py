# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.corpora import WikiCorpus
import os
import shutil
import subprocess



def xml_to_txt(xml_file_path, txt_file_path):
    txt_file_dir = os.path.dirname(txt_file_path)
    txt_file_name = os.path.basename(txt_file_path)
    if "_繁体" not in txt_file_name:
        txt_file_name = txt_file_name.split(".")[0].join("_繁体.txt")
    txt_file_path = os.path.join(txt_file_dir , txt_file_name)
    space = " "
    i = 0
    output = open(txt_file_path, 'w', encoding='utf-8')
    wiki = WikiCorpus(xml_file_path, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        s = space.join(text) + "\n"
        output.write(s)
        i = i + 1
        if (i % 10000 == 0):
            print("Save" + str(i) + "articles")
    output.close()
    print("Finished Saved" + str(i) + "articles")



def traditional_to_simplified(opencc_dir, txt_traditional_path, txt_simplified_path):
    traditional_file_name = os.path.basename(txt_traditional_path)  # 繁体文件名称
    simplified_file_name = os.path.basename(txt_simplified_path)  # 简体文件名称
    # 复制文件
    source_path = txt_traditional_path
    target_path = os.path.join(opencc_dir , traditional_file_name)
    shutil.copyfile(source_path, target_path)  # 把繁体文件复制到 opencc 安装目录下
    # 执行 opencc 命令，把繁体转换为简体
    cmd1 = opencc_dir.split("\\")[0]
    cmd2 = "cd " + opencc_dir.split("\\")[-1]
    cmd3 = "opencc -i " + traditional_file_name + " -o " + simplified_file_name + " -c t2s.json"
    cmd = cmd1 + " && " + cmd2 + " && " + cmd3
    sub = subprocess.Popen(cmd, shell=True)
    sub.wait()
    # 复制文件
    source_path = os.path.join(opencc_dir , simplified_file_name)
    target_path = os.path.join(os.path.dirname(txt_traditional_path), simplified_file_name)
    shutil.copyfile(source_path, target_path)  # 把繁体文件复制到 opencc 安装目录下
    # 删除文件
    os.remove(os.path.join(opencc_dir , traditional_file_name))
    os.remove(os.path.join(opencc_dir , simplified_file_name))
    print("Finished 繁体转换为简体")





opencc_dir = r"D:\opencc-1.0.1-win64"
data_dir = r"C:\Users\lenovo\Desktop\data\维基百科"
xml_path = os.path.join(data_dir, "zhwiki-20180801-pages-meta-history4.xml-p2375451p2771086.bz2")
txt_traditional_path = os.path.join(data_dir, "wiki_繁体.txt")
txt_simplified_path = os.path.join(data_dir, "wiki_简体.txt")


if __name__ == '__main__':
    # 解码xml文件，写入到txt文件，是繁体字
    xml_to_txt(xml_path, txt_traditional_path)
    # 把繁体字转换为简体字
    traditional_to_simplified(opencc_dir, txt_traditional_path, txt_simplified_path)


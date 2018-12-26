# -*- coding: utf-8 -*-

import requests
import hashlib
import time
import random
import string
from urllib.parse import quote



def get_md5(sign_before_str, encode):
    """
    功能：计算 md5
    参数：
        sign_before_str：计算 md5 之前的字符串
        encode：字符编码方式
    返回值：md5 的值(str类型)
    """
    m = hashlib.md5(sign_before_str.encode(encode))
    # 将得到的MD5值所有字符转换成大写
    return m.hexdigest().upper()



def req_sign(app_key, params, encode):
    """
    功能：计算签名
    参数：
        app_key：app_key
        params：其他请求参数
        encode：字符编码方式
    返回值：签名(str类型)
    """
    sign_before = ''
    # 要对key排序再拼接
    for key in sorted(params):
        # 键值拼接过程value部分需要URL编码，URL编码算法用大写字母，例如%E8。quote默认大写。
        sign_before += '{}={}&'.format(key, quote(params[key], safe=''))
    # 将应用密钥以app_key为键名，拼接到字符串sign_before末尾
    sign_before += 'app_key={}'.format(app_key)
    # 对字符串 sign_before 进行 MD5 运算，得到接口请求签名
    sign = get_md5(sign_before, encode)
    return sign



def get_params(app_id, app_key, params, encode):
    """
    功能：获取请求参数
    参数：
        app_id：app_id
        app_key：app_key
        params：其他请求参数
        encode：字符编码方式
    返回值：全部请求参数(字典类型)
    """
    # 把 app_id 加到 params 中
    params["app_id"] = app_id
    # 请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效）
    t = time.time()
    time_stamp = str(int(t))
    params["time_stamp"] = time_stamp
    # 请求随机字符串，用于保证签名不可预测
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 10))
    params["nonce_str"] = nonce_str
    # 获取签名
    sign = req_sign(app_key, params, encode)
    params["sign"] = sign
    return params



def calling_api(app_id, app_key, params, url, encode):
    """
    功能：调用腾讯AI开放平台的API
    参数：
        app_id：app_id
        app_key：app_key
        params：其他请求参数
        url：API地址
        encode：字符编码方式
    返回值：调用腾讯AI开放平台的API返回的数据(json类型)
    """
    # 获取请求参数
    params = get_params(app_id, app_key, params, encode)
    result = requests.post(url, data=params)
    data = result.json()["data"]
    return data


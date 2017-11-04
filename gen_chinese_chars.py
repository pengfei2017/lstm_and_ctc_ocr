#!/usr/bin/env python
# encoding=utf-8
# Compatibility imports
import random

"""
生成常见的中文字符

"""


def GBK2312():
    head = random.randint(0xb0, 0xf7)
    print(f'{head:x}')
    body = random.randint(0xa1, 0xfe)
    val = f'{head:x}{body:x}'
    str = bytes.fromhex(val).decode('gb2312')
    return str


def gen_chinese_chars():
    chars = []
    for i in range(0xb0, 0xf8):
        for j in range(0xa1, 0xff):
            val = f'{i:x}{j:x}'
            # print(val)
            try:
                str = bytes.fromhex(val).decode('gb2312')
            except Exception as e:
                print(e)
            else:
                # print(str)
                chars.append(str)
    return chars


if __name__ == '__main__':
    chinese_chars = gen_chinese_chars()
    print("".join(chinese_chars))
    print('简体中文的数目为', len(chinese_chars))

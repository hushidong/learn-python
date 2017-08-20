#!/usr/bin/env python3
#_*_coding: utf-8 _*_

import sys
print('default encoding is:',sys.getdefaultencoding())

def bytestounicodeb(byteobj):
    str=byteobj.decode()#以默认编码格式解析字节信息为字符
    ucode=hex(ord(str)) #得到字符的整数编码并转化为16进制数表示的字符串
    print("char's bytes is",byteobj)
    print('char is',str)
    print("char's code is %d, base=10" % ord(str))
    print("char's code is %s, base=16" % ucode)
    print("char's code is %s, base=2" % bin(ord(str)))

def bytestounicode(byteobj):#验证utf-8格式
    str=byteobj.decode()#以默认编码格式解析字节信息为字符
    str0=bin(byteobj[0])#得到第一个字节的二进制数构成的字符串
    nbytes=len(byteobj)#也可以用下面注释这一段来判断
    # nbytes=0
    # if len(str0)<10: #单字节字符高位的0不给出，所以加上0b后字符串位数会小于10
    #     strucode=str0
    # else:
    #     for i in range(2,10):
    #         if str0[i]=="0":
    #             break
    #         else:
    #             nbytes+=1
    if nbytes>1:
        strucode='0b'+str0[2+nbytes:]#二进制字符前面加'0b'或者不加对于类型转换没有影响，这里为了显示效果加上它
        for i in range(1,nbytes):
            strucode+=bin(byteobj[i])[4:]#不是第一个字节则取0b10后的字符，即从第5个字符(第4个索引)开始
    else:
        strucode=str0
    print("char's bytes is",byteobj)
    print('char is',str)
    print("char's code is %d, base=10" % int(strucode,base=2))
    print("char's code is %s, base=16" % hex(int(strucode,base=2)))
    print("char's code is %s, base=2" % strucode)

def chartounicode(char):
    print('char is',char)
    bytestounicode(char.encode())

def chartounicodeb(char):
    print('char is',char)
    bytestounicodeb(char.encode())

def unicodetochar(numobj):#利用二进制字符串的操作来得到utf8格式编码
    if numobj < 0x007f:#确定字节数
        nbytes=1
    elif numobj < 0x07ff:
        nbytes=2
    elif numobj < 0xffff:
        nbytes=3
    elif numobj < 0x10ffff:
        nbytes=4
    else:
        print("error")
    ucodestr=bin(numobj)
    blist=[]
    for i in range(nbytes):#处理得到各个字节的信息
        start=len(ucodestr)-6-6*i #不处理最前面一个字节时，取6个字符
        end=len(ucodestr)-i*6
        if i==nbytes-1: start=2 #处理最前面一个字节时，取去掉0b剩下的字符
        #print(start,end,ucodestr[start:end])
        if i==nbytes-1:
            strheader='11111111'[:nbytes]
            strheader+='00000000'[:8-nbytes-(end-start)]
            # for j in range(nbytes): #上述两句用for循环也可以
            #     strheader+='1'
            # numzero=8-nbytes-(end-start)
            # for j in range(numzero):
            #     strheader+='0'
            blist.insert(0,strheader+ucodestr[start:end])#存入blist列表中
        else:
            blist.insert(0,'10'+ucodestr[start:end])
    strhex="".join(hex(int(elem,base=2))[2:] for elem in blist)#这里利用了join方法连接字符串
    print("char's code is %d, base=10" % numobj)
    print("char's code is %s, base=16" % hex(numobj))
    print("char's code is %s, base=2" % ucodestr)
    print("char's byte string is",strhex)
    print("char's bytes is",bytes.fromhex(strhex))#fromhex的参数只要是由2个16进制的数的字符串构成即可
    print("char is",bytes.fromhex(strhex).decode())


def unicodetocharb(numobj):#利用整数的位的操作来得到utf8格式编码
    if numobj < 0x007f:#确定字节数
        nbytes=1
        ref=0 #00000000用于1字节的位或操作
    elif numobj < 0x07ff:
        nbytes=2
        ref=0xc080 #1100000010000000用于2字节的位或操作
    elif numobj < 0xffff:
        nbytes=3
        ref=0xe08080 #111000001000000010000000用于3字节的位或操作
    elif numobj < 0x10ffff:
        nbytes=4 #11110000100000001000000010000000用于4字节的位或操作
        ref=0xf0808080
    else:
        print("error")
    res=ref
    src=numobj
    for i in range(nbytes):#遍历nbytes个字节，顺序是从后往前
        if i==nbytes-1:#处理最前面一个字节时剩下的位全部取出用于位或
            a=src<<(i*8)
            res=res |a
        else:
            a=src & 0x3f #不处理最前面一个字节时，取6位用于位或
            src=src>>6 #源整数中，6位取出后直接丢弃
            a=a<<(8*i) #把取出的6为放到i字节上用于位或
            res=res | a
    #print ("%x" %res)
    strhex=("%x" %res)#这里利用了printf方式的字符串转换
    print("char's code is %d, base=10" % numobj)
    print("char's code is %s, base=16" % hex(numobj))
    print("char's code is %s, base=2" % bin(numobj))
    print("char's byte string is",strhex)
    print("char's bytes is",bytes.fromhex(strhex))
    print("char is",bytes.fromhex("%x" %res).decode())


print("")
bytestounicodeb("约".encode())
print("")
bytestounicode("约".encode())

print("")
bytestounicodeb("A".encode())
print("")
bytestounicode("A".encode())


print("")
bytestounicodeb('\u00b1'.encode()) #字符字面常量，参见python3.6.2.chm 2.4. Literals
print("")
bytestounicode('\u00b1'.encode())


print("")
bytestounicodeb('\U000E0051'.encode()) #字符字面常量，参见python3.6.2.chm 2.4. Literals
print("")
bytestounicode('\U000E0051'.encode())

print("")
unicodetochar(20013)
unicodetocharb(0x4e2d)#不同进制的输入都是可以的

chartounicode("国")
chartounicodeb("国")
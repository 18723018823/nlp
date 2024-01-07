import jieba
import json

# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open(r'baidu_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords
# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    info_dict = json.loads(sentence)
    print("正在分词")
    cut_list = jieba.lcut(info_dict['content'] + info_dict['title'])
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in  cut_list:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
# 给出文档路径
inputs = open(r"train.txt", 'r', encoding='UTF-8')
outputs = open(r"result.txt", 'w', encoding='UTF-8')
# 将输出结果写入result.txt中
for line in inputs:
    line_seg = seg_depart(line)
    outputs.write(line_seg + '\n')
outputs.close()
inputs.close()
print("删除停用词和分词成功！！！")
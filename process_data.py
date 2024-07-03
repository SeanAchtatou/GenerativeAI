import csv

corpus_movie_conv = 'data/movie_conversations.txt'
corpus_movie_lines = 'data/movie_lines.txt'
max_len = 25


with open(corpus_movie_conv, 'r') as c:
    conv = c.readlines()

with open(corpus_movie_lines, 'r') as l:
    lines = l.readlines()



lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]


def remove_punc(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char  # space is also a character
    return no_punct.lower()


data = []
for con in conv:
    ids = eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []
        
        if i==len(ids)-1:
            break
        
        first = remove_punc(lines_dic[ids[i]].strip())      
        second = remove_punc(lines_dic[ids[i+1]].strip())
        data.append([first,second])
        

filecsv = open("data/data.csv", "w", newline="")
filecsvwrite = csv.writer(filecsv)
filecsvwrite.writerows(data)
filecsv.close()


import numpy as np
import re
import pandas as pd
import json
def line_finder():
    '''
    파일에서 필요한 부분만 추출
    
    '''
    filename = r"C:\Users\MarketPoint\Downloads\log.txt"
    
    with open(filename,'rt') as f:
        Lines = f.readlines()
    for i, l in enumerate(Lines):
        matches = re.findall("val acc epoch:", l)
        if matches:
            print((Lines[i-6].strip() + l).strip()) #### 6줄 위의 line도 같이 출력


def line_remover():
    '''
    파일에서 특정 문자열이 포함된 line을 제거
    
    
    '''
#     input_file = r"D:\MathWordProblemSolving\SVAMP-arkilpatel\data\ko_en5\math23k_en_ko_raw.csv"
#     output_file = r"D:\MathWordProblemSolving\SVAMP-arkilpatel\data\ko_en5\math23k_en_ko.csv"


#     input_file = r"D:\MathWordProblemSolving\SVAMP-arkilpatel\data\ko_en5\ape210k_train_en_ko_raw.csv"
#     output_file = r"D:\MathWordProblemSolving\SVAMP-arkilpatel\data\ko_en5\ape210k_train_en_ko.csv"

    input_file = r"D:\MathWordProblemSolving\SVAMP-arkilpatel\data\ko_en4_new\train.csv"
    output_file = r"D:\MathWordProblemSolving\SVAMP-arkilpatel\data\ko_en4_new\train_.csv"

    with open(input_file,'rt',encoding='utf8') as f:
        Lines = f.readlines()
    
    
    bad_counter = 0
    good_counter = 0
    result_all = []
    for line in Lines:

        if line.find(r'((())/(()))') >=0:
            bad_counter = bad_counter+1

        elif line.find('\\') >=0:   # back slash가 있는 latex 코드.
            bad_counter = bad_counter+1
        
        elif line.find('계산:') >=0:
            bad_counter = bad_counter+1        
        else:
            result_all.append(line)
            good_counter = good_counter +1
    
    



    #random.shuffle(result_all)
    with open(output_file,'wt',encoding='utf8') as f:
        for result in result_all:
            f.write(result.replace('"',' '))   # result 끝에 줄바꿈 포함.


    print('good: ',good_counter)
    print('bad: ',bad_counter)

def find_patten():
    # number + 숫자 + 공백아닌 뭔가. 예. 'number1학교' 이런 것. 'number2 학교'와 같이 공백이 있는 것은 아니고...
    
    s = 'number3 number2학교 number0'
    
    matches = re.findall(r"number\d{1}[^ ]",s)
    
    print('sample: ', matches)
    
    input_file = r"D:\MathWordProblemSolving\SVAMP-arkilpatel\data\ko_en4_new\train.csv"
    
    Lines = pd.read_csv(input_file)
        
    for line in Lines['Question']:
            
        matches = re.findall(r"number\d{1}[^ ]",line)
        
        
        if matches:
            print(line)
            exit()

def json_loader():
    json_filename = r"C:\Users\BRAIN\Downloads\ape210k_train_en_ko.json"
    json_output = r"C:\Users\BRAIN\Downloads\xxx.json"
    
    
    data = json.load(open(json_filename,'r',encoding='UTF8'))
    print(type(data))
    with open(json_output, "wt",encoding='UTF8') as json_file:
    
        json.dump(data, json_file,ensure_ascii=False, indent=4)


def file_line_shuffle():
    filename = r"D:\MathWordProblemSolving\SVAMP-arkilpatel\data\ko_en6_new\arithmetic_data_ver2.csv"
    output_file = r"D:\MathWordProblemSolving\SVAMP-arkilpatel\data\ko_en6_new\arithmetic_data_ver2_shuffle.csv"
    
    
    
    with open(filename,'rt',encoding='utf8') as f:
        Lines = f.readlines()    # list

    head = Lines[0]
    Lines = Lines[1:]
    
    random.shuffle(Lines)
    print(type(Lines))

    with open(output_file,'wt',encoding='utf8') as f:
        f.write(head)
        for line in Lines:
            f.write(line)   # result 끝에 줄바꿈 포함.
if __name__ == '__main__':
    line_finder()
    #line_remover()
    #find_patten()
    #json_loader()
    #file_line_shuffle()

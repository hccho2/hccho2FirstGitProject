'''
windows 10 mecab설치

1. pip install eunjeon    ---> https://github.com/koshort/pyeunjeon

2. 사용자 사전 설치:
    - https://velog.io/@kjyggg/%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0-Mecab-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-A-to-Z%EC%84%A4%EC%B9%98%EB%B6%80%ED%84%B0-%EB%8B%A8%EC%96%B4-%EC%9A%B0%EC%84%A0%EC%88%9C%EC%9C%84-%EB%93%B1%EB%A1%9D%EA%B9%8C%EC%A7%80
    - 2개 파일을 다운 받아, c:\mecab에 압축을 푼다.
    mecab-ko-msvc
    https://github.com/Pusnow/mecab-ko-msvc/releases/download/release-0.9.2-msvc-3/mecab-ko-msvc-x64.zip
    
    
    mecab-ko-dic-msvc
    https://github.com/Pusnow/mecab-ko-dic-msvc/releases/download/mecab-ko-dic-2.1.1-20180720-msvc/mecab-ko-dic-msvc.zip
    
    
3 사용자 사전 update:
    - powershell에서     
    C:\mecab> tools\add-userdic-win.ps1     -----> 설치 에러: http://blog.naver.com/PostView.nhn?blogId=bluesketch21&logNo=221383264763&categoryNo=4&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView

'''


from eunjeon import Mecab

dicpath = r'C:\mecab\mecabrc'


mecab = Mecab(dicpath) 
s = '우리는 불교의 정신을 존중해야 한다.'
s = '아미코젠 주가는 어떻게 전망하세요?'
#s = '불교의'

print(mecab.morphs(s))

print(mecab.pos(s))
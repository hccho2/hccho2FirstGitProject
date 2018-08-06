# -*- coding: utf-8 -*-
import json
import numpy as np
from gtts import gTTS


import os
import pyglet   # AVbin이라는 프로그램을 성치해야 함(http://avbin.github.io/AVbin/Download.html)

def exit_callback(dt):
    pyglet.app.exit()


filename = "hello.mp3"
tts = gTTS(text='안녕하세요. 조희철이라고 합니다. 현종석친구에요', lang='ko', slow=False)

tts.save(filename)


#os.system(filename)  # media player 실행 후 닫히지 않음. 반복해서 사용할 수 없음.


song = pyglet.media.load(filename)
song.play()

pyglet.clock.schedule_once(exit_callback , song.duration)  # 이게 있어야 ,pyglet.app.run()이 종료됨


pyglet.app.run()

print("Done")
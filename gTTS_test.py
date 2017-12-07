# -*- coding: utf-8 -*-
import json
import numpy as np
from gtts import gTTS

from pygame import mixer
import os
import pyglet   # AVbin이라는 프로그램을 성치해야 함(http://avbin.github.io/AVbin/Download.html)

filename = "hello.mp3"
tts = gTTS(text='안녕하세요. 조희철이라고 합니다. 박일재 친구에요', lang='ko', slow=False)

tts.save(filename)


#os.system(filename)  # media player 실행 후 닫히지 않음. 반복해서 사용할 수 없음.


song = pyglet.media.load(filename)
song.play()
pyglet.app.run()


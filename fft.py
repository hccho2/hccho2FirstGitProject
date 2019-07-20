# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
x = np.arange(128)
t = np.sin(np.arange(16))
sp128 = np.fft.fft(t,128)  # output size 지정
sp32 = np.fft.fft(t,32)
sp16 = np.fft.fft(t,16)
sp8 = np.fft.fft(t,8)
sp4 = np.fft.fft(t,4)

plt.plot(x,np.abs(sp128),label='128')
plt.plot(x[::4],np.abs(sp32),label='32')
plt.plot(x[::8],np.abs(sp16),label='16')
plt.plot(x[::16],np.abs(sp8),label='8')
plt.plot(x[::32],np.abs(sp4),label='4')
plt.legend(loc='upper center')
plt.show()


sp8_ = np.fft.fft(t[:8],8)  # resolution이 낮아지면 뒤쪽 data를 버린다.
plt.plot(x[::16],np.abs(sp8),label='8')
plt.plot(x[::16],np.abs(sp8_),label='8_')
plt.legend(loc='upper center')
plt.show()


sp32_ = np.fft.fft(np.concatenate([t,np.zeros(16)]),32) # resulution을 높이기 위해서는 뒤쪽에 zoro padding을 한다.
plt.plot(x[::4],np.abs(sp32),label='32')
plt.plot(x[::4],np.abs(sp32_),label='32_')
plt.legend(loc='upper center')
plt.show()
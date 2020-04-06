# coding: utf-8

import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import os
import numpy as np

def plot_line(xs, ys_population, save_dir):
    max_colour = 'rgb(0, 132, 180)'
    mean_colour = 'rgb(0, 172, 237)'
    std_colour = 'rgba(29, 202, 255, 0.2)'

    ys = np.array(ys_population)
    ys_min = np.min(ys, axis=1)
    ys_max = np.max(ys, axis=1)
    ys_mean = np.mean(ys, axis=1)
    ys_std = np.std(ys, axis=1)
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    # 최대
    trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')

    # 1-sigma
    trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color='whitesmoke'), name='+1 Std. Dev.', showlegend=False)
    
    # fillcolor는 선과 선사이을 채우는 역할
    trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    
    # 1- sigam
    trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color='whitesmoke'), name='-1 Std. Dev.', showlegend=False)
    
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
      'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
      'layout': dict(title='Rewards',
                     xaxis={'title': 'Step'},
                     yaxis={'title': 'Average Reward'})
    }, filename=os.path.join(save_dir, 'rewards.html'), auto_open=False)
    

x = [3,5,9]
y = np.random.randn(len(x),10)

plot_line(x,y,'./result')

print('Done')
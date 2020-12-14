#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
time:
"""

# fw =  open('frameAvgData.txt','w')     #设置文件对象
#
# with open('SaveFullT_0011.txt',"r") as f:    #设置文件对象
#     lineIndex = 0
#     frameSum = 0
#     line = f.readline()  # 调用文件的 readline()方法
#     while line:
#         if lineIndex % 120 > 54 and lineIndex % 120 < 64:
#             data = line.split(' ')
#             results = data[75:84]
#             frameSum += sum(list(map(float, results)))
#         if lineIndex % 120 == 64:
#             frameavg =  frameSum /81
#             fw.write('%.2f' % frameavg)
#             fw.write('\n')
#             frameSum = 0
#         line = f.readline()
#         lineIndex += 1
#
# fw.close()

#提取中心点温度数据
fw =  open('CenterPointData_1mdhnewCamera.txt','w')     #设置文件对象

with open(r'D:\catData\image_1mdhnewCamera.txt',"r") as f:    #设置文件对象
    lineIndex = 1
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        # if lineIndex % 4 == 3:  #原厂demo格式
        if lineIndex % 3 == 1:  #自研格式
            fw.write(line.split(' ')[-1])
        line = f.readline()
        lineIndex += 1

fw.close()


# 画mat图
# import matplotlib.pyplot as plt
# import numpy as np
# rawdata = np.loadtxt(r'CenterPointData.txt')
# plt.plot(rawdata[:])
# plt.show()


##画SVG图
# import pygal
# line_chart = pygal.Line()
# line_chart.title = 'Browser usage evolution (in %)'
# line_chart.x_labels = map(str, range(0, len(data[:1000])))
# line_chart.add('Firefox', data[:1000])
# line_chart.render_to_file('Hello_line_chart.svg')

#
###画html文件
# import numpy as np
# import plotly.offline as of
# import plotly.graph_objs as go
# import pandas as pd
#
#
#
# filename = r'E:\0321\CenterPointData_distan.txt'
# # filename = r'CenterPointData.txt'
# temperatureData = np.loadtxt(filename)
#
# dataLength = len(temperatureData)
#
# ##pandas
# rawPdData = pd.DataFrame(temperatureData)
#
#
# rawPdDataCondition = rawPdData[rawPdData> 32]
# stdValuePd = rawPdDataCondition.std()
# meanValuePd = rawPdDataCondition.mean()
# maxValuePd = rawPdDataCondition.max()
# minValuePd = rawPdDataCondition.min()
# maxindex= rawPdDataCondition.idxmax()[0]
# rawPdtoNp = rawPdDataCondition[0].to_numpy()
# print('stdValuePd {}, meanValuePd {}, maxValuePd {}, minValuePd {}'.format(
#     stdValuePd, meanValuePd, maxValuePd, minValuePd))
#
#
# # dataDropNaN = rawPdDataCondition.dropna(how='any')
# # rawPd11 = dataDropNaN.to_numpy()
# # npstd = np.std(rawPd11)
# # npmean = np.mean(rawPd11)
# # npmmax = np.max(rawPd11)
# # npmin = np.min(rawPd11)
# # print('npstd {}, npmean {}, npmmax {}, npmin {}'.format(npstd, npmean, npmmax, npmin))
#
# #
# ##
#
# # #分组取均值
# # meanUint = 15
# # clipLength = dataLength//meanUint
# # dataClip = temperatureData[:clipLength * meanUint]
# # meanDataPerSeconds = np.mean(np.reshape(dataClip, (-1,meanUint)), axis =  1)
# # ###
#
# tempStd = np.std(temperatureData)
# print('tempStd :{}'.format(tempStd))
#
# # # 差值
# # tempDif = np.diff(temperatureData)
# # temperatureData = tempDif
# # minvalue = np.amin(np.abs(temperatureData[np.nonzero(temperatureData)]))
# # print(minvalue)
#
# # 机芯温度
# filename = r'E:\0321\CameraTemperature_distan.txt'
# camTempData = np.loadtxt(filename)[:]
#
# # 快门帧数温度
# filename = r'E:\0321\shutterFlag_distan.txt'
# camAdjIndexs = np.array(np.loadtxt(filename)[:, 0], dtype=np.int32)
# # camAdjTempData = temperatureData[camAdjIndexs[np.where(camAdjIndexs<54000)]]
# camAdjTempData = temperatureData[camAdjIndexs]
#
#
# #统计值：输入参数 dataLength 和 temperatureData
# defineValue = [35.0]
# xPoints = list(range(dataLength))
#
# maxValue = [np.amax(temperatureData)]
# maxXpoins = [np.where(temperatureData == maxValue)[0][0]]
#
# minValue = [np.amin(temperatureData)]
# minXpoins = [np.where(temperatureData == minValue)[0][0]]
#
# midValue = [np.median(temperatureData)]
# midXpoins = [np.where(temperatureData == midValue)[0][0]]
#
# meanValue = [np.mean(temperatureData)]
# print('meanValue :{}'.format(meanValue))
# # meanXpoins = [dataLength//2]
#
#
# # Create traces
# trace0 = go.Scatter(
#     x=xPoints,
#     dx=1,
#     y=rawPdtoNp,
#     # text=temperatureData,
#     # fill="tonexty",
#     marker=dict(size=3, color='skyblue', line=dict(width=1, color='skyblue'), ),  # cadetblue skyblue slateblue
#     mode='lines',
#     name='实际测量值'
# )
# traceObj = go.Scatter(
#     x=xPoints,
#     dx=1,
#     y=defineValue * dataLength,
#     # text=defineValue*dataLength,
#     # marker='^',  #正三角形
#     marker={'symbol': 4, 'size': 5},
#     mode='lines',
#     # mode='lines+markers',
#     name='期望温度值'
# )
# # 统计数值曲线
# traceMax = go.Scatter(
#     x=maxXpoins,
#     dx=1,
#     y=maxValue,
#     # text=maxValue *dataLength,
#     marker={'symbol': 201, 'size': 10},
#     # marker=dict(size=3, color='skyblue', line=dict(width=1, color='skyblue') ),
#     line=dict(color=('rgb(205, 12, 24)'), width=2, dash='dash'),
#     mode='markers',
#     name='最大值'
# )
# traceMin = go.Scatter(
#     x=minXpoins,
#     dx=1,
#     # y=minValue,
#     y=minValuePd,
#     # text=minValue,
#     marker={'symbol': 4, 'size': 10},
#     line=dict(color=('rgb(205, 12, 24)'), width=2, dash='dash'),
#     mode='markers',
#     name='最小值'
# )
#
# traceMean = go.Scatter(
#     # x=meanXpoins,
#     x=xPoints,
#     dx=1,
#     y=meanValue * dataLength,
#     # text=defineValue*dataLength,
#     # marker='^',  #正三角形
#     line=dict(color=('rgb(205, 12, 24)'), width=2, dash='dash'),
#     mode='lines',
#     name='测量平均值'
# )
#
# # 机芯温度
# traceCamTempData = go.Scatter(
#     # x=meanXpoins,
#     x=xPoints,
#     dx=1,
#     y=camTempData,
#     # text=defineValue*dataLength,
#     # marker='^',  #正三角形
#     line=dict(color='#ff7500', width=2),
#     mode='lines',
#     name='机芯温度'
# )
#
# # 快门帧数温度
# tracecamAdjTempData = go.Scatter(
#     x=camAdjIndexs,
#     dx=1,
#     y=camAdjTempData,
#     # text=minValue,
#     marker={'symbol': 117, 'size': 5, 'color': '#0aa344'},  # 117五角星
#     line=dict(color='#0aa344', width=2, dash='dash'),
#     mode='markers',
#     name='快门校正点'
# )
#
# # # 差值
# # traceDiff = go.Scatter(
# #     x=xPoints[1:],
# #     dx=1,
# #     y=tempDif,
# #     # text=temperatureData,
# #     # fill="tonexty",
# #     marker=dict(size=3, color='skyblue', line=dict(width=1, color='skyblue'), ),  # cadetblue skyblue slateblue
# #     mode='lines',
# #     name='实际测量差值'
# #     # opacity = 1
# # )
#
# # trace11 = go.Bar(
# #     x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
# #     y=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
# #     marker=dict(
# #         color=["#FF0000", "#00FF00"],
# #     ),
# #     width=0.005
# # )
# # data = [trace11]
#
#
#
# # data = [ trace0, traceObj,traceMax]
# data = [trace0, traceObj, traceMax, traceMin, tracecamAdjTempData, traceCamTempData]
# # data = [traceDiff, traceMax, traceMin, tracecamAdjTempData]
#
# # 随时间变化的一天中的订单需求量
# layout = go.Layout(title='大立机芯连续10小时温度测量值',
#                    titlefont={'size': 20},
#                    xaxis={
#                        'title': '帧序号(帧率：15FPS)',
#                        # 'type':'data', #'date'表示日期型坐标轴
#                        # 'autorange' : False,  #autorange：bool型或'reversed'，控制是否根据横坐标对应的数据自动调整坐标轴范围，默认为True
#                        'showgrid': False,  # 是否显示网格
#                        'zeroline': True,  # 是否显示基线,即沿着(0,0)画出x轴和y轴
#                        'titlefont': {
#                            'size': 15
#                        },
#                        'nticks': 24  # x轴最大刻度到24
#                    },
#                    yaxis={
#                        'title': '温度(°C)',
#                        'zeroline': True,  # 是否显示基线,即沿着(0,0)画出x轴和y轴
#                        'titlefont': {
#                            'size': 15
#                        }
#                    },
#                    # plot_bgcolor="#faff72",
#                    #  annotations=[
#                    # dict( x=2, y=5,
#                    #      xref='x', yref='y',
#                    #      text='dict Text',
#                    #      # 设置注释的字体参数
#                    #      font=dict(family='Courier New, monospace', size=16, color='#ffffff'),
#                    #      showarrow=True,
#                    #      # 设置显示箭头
#                    #      # 设置箭头的参数
#                    #      ax=20, ay=-30, align='center', arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#636363',
#                    #      # 设置注释的边框
#                    #      bordercolor='#c7c7c7', borderwidth=2, borderpad=4, bgcolor='#ff7f0e', opacity=0.8),]
#                    )
# fig = go.Figure(data=data, layout=layout)
# of.plot(fig, filename='dataset18.html')








##画html文件
# from  plotly.offline  import  init_notebook_mode,  iplot,  plot
# import  plotly.graph_objs  as  go
# import  pandas  as  pd
# import  numpy  as  np
# #在notebook中需要初始化，py文件中不用
# # init_notebook_mode ()
# # #产生500个数据点
# # N  =   500
# # x  =  np . linspace ( 0 ,   1 ,  N )#从 0 到 1 ，产生 500 个数据点
# # y  =  np . random . randn ( N )
#
# # N  =   500
# x  =  range(len(rawdata[:1000]))
# y  =  rawdata[:1000]
#
#
# df  =  pd . DataFrame ({ 'x' :  x , 'y' :  y })
# data = [ go . Scatter ( x =  df [ 'x' ], y  =  df [ 'y' ])]
# #在py文件中使用
# plot(data)
# # iplot ( data )


# import plotly.graph_objs as go
# import plotly.offline as pltoff
# from  plotly.offline  import  init_notebook_mode ,  iplot ,  plot
# trace1 = go.Scatter(
#     x=[1, 2, 3, 4, 5,
#        6, 7, 8, 9, 10,
#        11, 12, 13, 14, 15],
#     y=[10, 20, None, 15, 10,
#        5, 15, None, 20, 10,
#        10, 15, 25, 20, 10],
#     name = '<b>No</b> Gaps', # Style name/legend entry with html tags
#     connectgaps=True
# )
# trace2 = go.Scatter(
#     x=[1, 2, 3, 4, 5,
#        6, 7, 8, 9, 10,
#        11, 12, 13, 14, 15],
#     y=[5, 15, None, 10, 5,
#        0, 10, None, 15, 5,
#        5, 10, 20, 15, 5],
#     name = 'Gaps',
# )
#
# data = [trace1, trace2]
# fig = dict(data=data)
# plot(fig, filename='simple-connectgaps.html')


####画3d图
import plotly
import plotly.graph_objs as go
import numpy as np
from  plotly.offline  import  init_notebook_mode ,  iplot ,  plot
z = np.linspace(0, 10, 50)
x = np.cos(z)
y = np.sin(z)
label = list(range(len(x)))
new_label = [x//20 for x in label]
trace = go.Scatter3d(
   x = x, y = y, z = z,mode = 'markers', marker = dict(
      size = 12,
#      color = z, # set color to an array/list of desired values
      color = new_label, # set color to an array/list of desired values 给每个点用标签区别颜色
      colorscale = 'Viridis',
      opacity=0.8
      ),
      text= new_label, #给每个点添加标签
   )
layout = go.Layout(title = '3D Scatter plot',
	margin=dict(
			l=0,
			r=0,
			b=0,
			t=0
		),
		scene=dict(
		  xaxis=dict(title='年龄'),
		  yaxis=dict(title='消费分数(1-100)'),
		  zaxis=dict(title='年收入(k$)')#设置x,y,z轴的label
		))
fig = go.Figure(data = [trace], layout = layout)
plot(fig, filename='simple-connectgaps.html')
fig.show()

###三维图和子图
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import random
py.init_notebook_mode()

subplot3d1 = go.Scatter3d(
    x=[random.random() for n in range(100)],
    y=[random.random() for n in range(100)],
    z=[random.random()*10 for n in range(100)],
    mode='markers',
    marker=dict(size=8, color=z, colorscale='Viridis',opacity=0.5))

subplot3d2 = go.Surface(
    z=[[(x * x + y * y) for x in range(-100, 100)] for y in range(-100, 100)],
    opacity=1)

fig = tools.make_subplots(
    rows=1, cols=2, specs=[[{
        'is_3d': True
    }, {
        'is_3d': True
    }]])
fig.append_trace(subplot3d1, 1, 1)
fig.append_trace(subplot3d2, 1, 2)

py.iplot(fig)
import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数

fig1 = plt.figure()  # 创建一个绘图对象
ax = Axes3D(fig1)  # 用这个绘图对象创建一个Axes对象(有3D坐标)
fig2 = plt.figure()  # 创建一个绘图对象
ax = Axes3D(fig2)  # 用这个绘图对象创建一个Axes对象(有3D坐标)
plt.show()  # 显示模块中的所有绘图对象

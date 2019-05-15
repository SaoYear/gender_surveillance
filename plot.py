import matplotlib.pyplot as plt
import numpy as np




# cost = [0.6507555, 0.35089982, 0.17975791, 0.057270896, 0.03769348, 0.01659458, 0.0017871595, 0.0003113766, 0.0001475101, 6.925745e-05, 6.691998e-05]
# accuracy = [0.628125, 0.8070313, 0.83203125, 0.8403646, 0.84244794, 0.8434896, 0.85078126, 0.8518229, 0.85260415, 0.8497396, 0.85286456]
# label = ['Cost', 'Accuracy']
# x = [i*4 for i in range(len(cost))]
# y = []
# y.append(cost)
# y.append(accuracy)
# # y.append(y_conv3_1)
#
# color = ['cornflowerblue', 'fuchsia',  'coral', 'lime', 'blueviolet', 'forestgreen', 'black', 'red']
# linestyle = ['-', '--', '-.', ':']
# marker = ['v', '1', 'h', 'd', 'o', '*', 'x', '+']
#
#
#
# plt.title("Performance on test")
# plt.ylabel("Accuracy/Loss")
# plt.xlabel("Epochs")
#
# for id, data in enumerate(y):
#     # label.append("{}_{}".format("", (id+2)))
#     marker_style = dict(color=color[id], linestyle=linestyle[id % 4], marker=marker[id],
#                         markersize=10)
#     plt.plot(x, data, **marker_style)
# plt.legend(label)
# plt.grid()
# plt.show()

with open("LOSS LOG", "r") as f:
    loss = [float(item) for item in f.read().split(',')]
print(len(loss))

l = loss[130::70]
x = [i+16 for i in range(len(l))]

plt.title("YOLOv3 Loss from epoch 15")
plt.ylabel("Loss")
plt.xlabel("Epochs")
marker_style = dict(color='cornflowerblue', linestyle=':', marker='*', markersize=10)
plt.plot(x, l, **marker_style)
plt.grid()
plt.show()

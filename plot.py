import pandas as pd
from matplotlib import pyplot as plt

from path import fig_path, log_path, f1_report_path, event_type


def train_plot(history, epoch, optimizer_name):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    train_process = pd.DataFrame(history)
    
    # 损失函数
    plt.plot(epoch, train_process.loss, marker = '^', markevery = 5, color = 'k', label = "Train loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(fig_path + "/" + event_type + "_train_loss_{}_{}.png".format(optimizer_name, comment))
    plt.show()
    
    # 精度
    plt.plot(epoch, train_process.global_pointer_f1_score, marker = '^', markevery = 5, color = 'k',
             label = "Train acc")
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.legend()
    plt.savefig(fig_path + "/" + event_type + "_train_f1_{}_{}.png".format(optimizer_name, comment))
    plt.show()
    
    train_process.to_csv(log_path + "/" + event_type + "_train_{}_{}.csv".format(optimizer_name, comment))


def f1_plot(data, optimizer_name):
    pd_f1 = pd.DataFrame(data)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    
    plt.plot(pd_f1.epoch, pd_f1.f1, marker = '^', markevery = 5, color = 'k', label = "Val f1")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.legend()
    plt.savefig(fig_path + "/" + event_type + "_val_f1_{}_{}.png".format(optimizer_name, comment))
    plt.show()
    
    pd_f1.to_csv(log_path + "/" + event_type + "_val_f1_{}_{}.csv".format(optimizer_name, comment), encoding = 'utf-8')


comment = "FINAL"

# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/7/29 15:12
# @Author  : wanghd
# @note    :

from math import sqrt, ceil
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# sns.set(font="SimHei")


def df_box_plot(df, layout=None, sub_title='box_plot'):
    """数值型变量（单变量）分布情况"""
    plt.figure(1, figsize=(12, 8))
    for i, col in enumerate(df.columns):
        if layout is None:
            # 如果不指定 layout，则根据列数开方进行计算指定
            squared_layout = ceil(sqrt(df.shape[1]))
            plt.subplot(squared_layout, squared_layout, i + 1)
        else:
            plt.subplot(*layout, i + 1)
        plt.boxplot(df.loc[:, col])
        plt.title(col)
    plt.suptitle(sub_title)
    plt.show()


def df_barplot(df, layout=None, sub_title="bar_plot"):
    """离散型变量（单变量）分布情况"""
    plt.figure(1, figsize=(12, 6))
    for i, col in enumerate(df.columns):
        if layout is None:
            # 如果不指定 layout，则根据列数开方进行计算指定
            squared_layout = ceil(sqrt(df.shape[1]))
            plt.subplot(squared_layout, squared_layout, i + 1)
        else:
            plt.subplot(*layout, i + 1)
        count = df[col].value_counts()
        plt.bar(count.index, count.values, width=0.5)
        plt.title(col)
    plt.suptitle(sub_title)
    plt.show()


def df_pair_boxplot(df, x, y, hue, layout=None, figsize=(12, 16), sub_title='pair_boxplot'):
    """分组箱线图，横轴x(类别型), 纵轴为y（可以为多列，数值型）, 根据hue分组（类别型）
    即查看数值型变量 在不同类别分组条件下的 分布情况
    离散型（单变量）和数值型（单变量）"""
    plt.figure(1, figsize=figsize, dpi=300)
    for i, col in enumerate(y):
        if layout is None:
            # 如果不指定 layout，则根据列数指定行数，列数总为1
            plt.subplot(len(y), 1, i + 1)
        else:
            plt.subplot(*layout, i + 1)
        sns.boxplot(x, col, hue, df)
    plt.suptitle(sub_title)
    plt.show()


def df_pair_plot(df):
    """数值型变量的相关分析图， 会自动过滤非数值型列
    数值型（双变量）"""
    sns.pairplot(df)
    plt.show()


def plot_learning_curve(estimator, title, x, y, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.05, 1., 20),
                        verbose=0, plot=True):
    """
    画出data在某模型上的learning curve. 判断当前模型的状态：过拟合，欠拟合
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    ----------
    estimator : 学习器
    title : 图像的标题
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                         validation_scores_mean + validation_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, validation_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (
            validation_scores_mean[-1] - validation_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (validation_scores_mean[-1] - validation_scores_std[-1])
    return midpoint, diff


def plot_validation_curve(estimator, x, y, param_name: str, param_range=range(1, 5), cv: int = 5,
                          scoring: str = "accuracy", y_lim: Tuple[float, float] = (0.0, 1.1)):
    """绘制验证曲线，判断单个超参数的取值范围"""
    # param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        estimator, x, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(f"Validation Curve with {estimator.__class__.__name__}")
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.ylim(*y_lim)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    pass

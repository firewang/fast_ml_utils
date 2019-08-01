# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/7/29 10:01
# @Author  : wanghd
# @note    : 分类任务全流程搭建，eda 部分抽离作图函数，特征工程部分抽离特征处理函数（类），
# @note    : 模型调参部分抽离学习器，评估方法，参数字典

import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report as cr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from figure_utils import df_box_plot, df_barplot, df_pair_boxplot, df_pair_plot
from figure_utils import plot_learning_curve
from feature_engineer_utils import CustomDummifier, CustomEncoder, my_pipeline

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font="SimHei")


def get_data(filename, filepath=os.getcwd()):
    """读取初始数据"""
    full_path = os.path.join(filepath, filename)
    if os.path.exists(full_path):
        suffix = filename.split(".")[-1]
        if suffix == 'csv':
            df = pd.read_csv(full_path)
        else:
            df = pd.read_excel(full_path)
        return df
    else:
        raise FileNotFoundError


def split_data(df, label_col_name, test_size=0.2):
    label = df.pop(label_col_name)
    train_x, test_x, train_y, test_y = train_test_split(df, label, random_state=11, stratify=label, test_size=test_size)
    # 重构row index
    train_x.reset_index(drop=True, inplace=True)
    test_x.reset_index(drop=True, inplace=True)
    train_y.reset_index(drop=True, inplace=True)
    test_y.reset_index(drop=True, inplace=True)
    return train_x, test_x, train_y, test_y


def eda_data(df):
    """初步的数据探索"""
    # 先查看数据基本情况
    df = df.replace("unknown", np.nan)

    # 缺失比例，均值，标准差，分位数等
    df_info = pd.DataFrame(df.isnull().sum(axis=0), columns=["nan_nums"])
    df_info.loc[:, 'nan_percent'] = df_info.loc[:, 'nan_nums'] / df.shape[0]
    df_info.loc[:, 'nunique'] = df.apply(pd.Series.nunique)  # 各列不同值的数量
    df_info = pd.merge(df_info, df.describe().T, left_index=True, right_index=True, how='left')
    print(df_info)

    # 初步数据探索
    # 数值型单变量分布
    df_box_plot(df.loc[:, ['age', 'balance', 'day']], sub_title='')
    # 离散型单变量分布
    df_barplot(df, sub_title='')
    # 离散型+ 数值型 单变量分布
    df_pair_boxplot(df, 'housing', ['age', ], 'y', figsize=(12, 16))
    # 数值型双变量相关性
    df_pair_plot(df.loc[:, ['age', 'balance', 'day', 'job', 'y']])
    return None


def basic_data_treatment(df):
    """初步数据处理（预处理）"""
    # 类别型热编码
    dummy = CustomDummifier(cols=['marital', 'contact', 'job', "poutcome", 'month'])
    # df = dummy.fit_transform(df)
    # 序列型数值编码
    label_encoder = CustomEncoder(col='education', ordering=['unknown', "primary", "secondary", 'tertiary'])
    # df = label_encoder.fit_transform(df)

    for col in ["default", "housing", 'loan']:
        df.loc[:, col] = df.loc[:, col].map({"no": 0, "yes": 1})

    uf_pipeline = my_pipeline(["dum", 'encoder'], [dummy, label_encoder])
    df = uf_pipeline.fit_transform(df)
    # pd.set_option('display.max_columns', 30)
    # print(df)
    return df


def deep_data_treatment(df):
    """进一步的特征工程：特征选择，特征构造"""
    return df


def basic_model_selection(basic_models=None, x=None, y=None, scoring='roc_auc', cv=5):
    """初步的算法筛选，确定一个或者几个基学习器进行下一步调参"""
    if basic_models is None:
        all_models = [linear_model.LogisticRegression(),
                      KNeighborsClassifier(),
                      DecisionTreeClassifier(),
                      # SVC(),
                      AdaBoostClassifier(),
                      AdaBoostClassifier(learning_rate=0.5),
                      AdaBoostClassifier(base_estimator=linear_model.LogisticRegression()),
                      RandomForestClassifier(),
                      GradientBoostingClassifier()]
    else:
        all_models = basic_models
    cv_scores = []
    for basic_model in all_models:
        cv_scores.append(cross_val_score(basic_model, X=x, y=y, scoring=scoring, cv=cv))
    cv_score_df = pd.DataFrame(cv_scores).T  # 获得各个学习器每折的score
    cv_score_df.columns = [basic_model.__class__.__name__ for basic_model in all_models]  # 将列名命名为学习器名
    cv_score_df.index = [f"cv_{cv_round + 1}" for cv_round in cv_score_df.index]  # index命名为cv_{index_round}
    cv_score_df = pd.concat([cv_score_df, cv_score_df.describe()], axis=0)  # 加入各轮cv的scores的统计信息
    print(cv_score_df)
    return cv_score_df


def model_param_tuning_lr(train_x, train_y, test_x, test_y, scoring='roc_auc'):
    """模型调参"""
    grid_search_param = [{
        'penalty': ["l1", 'l2'],
        'C': [1, 10],
        'solver': ["liblinear"],
    },
        # {
        #     'penalty': ['l2'],
        #     'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        #     'solver': ["lbfgs"],
        # }
    ]
    grid_search_clf = GridSearchCV(linear_model.LogisticRegression(tol=1e-6, max_iter=1000),
                                   grid_search_param,
                                   cv=10,
                                   n_jobs=-1,
                                   scoring=scoring)
    grid_search_clf.fit(train_x, train_y)
    print(grid_search_clf.best_params_)
    # print(grid_search_clf.best_estimator_.coef_)  # 逻辑回归的各项（特征）系数 == 特征重要性
    feature_importance = pd.DataFrame(grid_search_clf.best_estimator_.coef_.reshape(-1, 1),
                                      columns=['Feature_importance'],
                                      index=train_x.columns)
    all_importances = abs(feature_importance.loc[:, 'Feature_importance']).sum()
    # 特征重要型占比
    feature_importance.loc[:, "feature_importance_percent"] = feature_importance.loc[:,
                                                              'Feature_importance'] / all_importances
    # 百分比都表示为 正 值
    feature_importance.loc[:, 'normalized_importance'] = feature_importance.feature_importance_percent.apply(abs)
    # 输出百分比格式列
    feature_importance.loc[:, 'normalized_importance_percentile'] = feature_importance.normalized_importance.apply(
        lambda x: "{:.2%}".format(x)
    )
    print("{:*^30}".format("特征重要性排序"))
    print(feature_importance.sort_values(by='Feature_importance', ascending=False))
    feature_importance.Feature_importance.sort_values(ascending=False).plot(kind='barh')
    plt.show()

    print("{:*^30}".format("最佳模型评估效果"))
    print(cr(test_y, grid_search_clf.best_estimator_.predict(test_x)))

    print("{:*^30}".format("最佳模型auc值"))
    print(roc_auc_score(test_y, grid_search_clf.best_estimator_.predict_proba(test_x)[:, 1]))
    return grid_search_clf.best_estimator_


def model_param_tuning_gbdt(train_x, train_y, test_x, test_y, scoring='roc_auc'):
    """GBDT模型调参"""
    grid_search_param = [{
        'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5, 0.8],
        'n_estimators': range(20, 101, 10),
        'max_depth': range(3, 22, 2),
        'min_samples_split': range(100, 801, 200),
        'min_samples_leaf': range(60, 101, 10)
    },
    ]
    grid_search_clf = GridSearchCV(GradientBoostingClassifier(tol=1e-6),
                                   grid_search_param,
                                   cv=10,
                                   n_jobs=-1,
                                   scoring=scoring)
    grid_search_clf.fit(train_x, train_y)
    print(grid_search_clf.best_params_)
    feature_importance = pd.DataFrame(grid_search_clf.best_estimator_.feature_importances_.reshape(-1, 1),
                                      columns=['Feature_importance'],
                                      index=train_x.columns)
    all_importances = abs(feature_importance.loc[:, 'Feature_importance']).sum()
    # 特征重要型占比
    feature_importance.loc[:, "feature_importance_percent"] = feature_importance.loc[:,
                                                              'Feature_importance'] / all_importances
    # 百分比都表示为 正 值
    feature_importance.loc[:, 'normalized_importance'] = feature_importance.feature_importance_percent.apply(abs)
    # 输出百分比格式列
    feature_importance.loc[:, 'normalized_importance_percentile'] = feature_importance.normalized_importance.apply(
        lambda x: "{:.2%}".format(x)
    )
    print("{:*^30}".format("特征重要性排序"))
    print(feature_importance.sort_values(by='Feature_importance', ascending=False))
    feature_importance.Feature_importance.sort_values(ascending=False).plot(kind='barh')
    plt.show()

    print("{:*^30}".format("最佳模型评估效果"))
    print(cr(test_y, grid_search_clf.best_estimator_.predict(test_x)))

    print("{:*^30}".format("最佳模型auc值"))
    print(roc_auc_score(test_y, grid_search_clf.best_estimator_.predict_proba(test_x)[:, 1]))
    return grid_search_clf.best_estimator_


if __name__ == '__main__':
    data = get_data("train_set.csv")
    data.pop("ID")
    test_data = get_data('test_set.csv')
    test_id = test_data.pop("ID")
    # print(data.head())

    df = basic_data_treatment(data)
    test_df = basic_data_treatment(test_data)
    # eda_data(data)

    # 算法选择
    train_x, test_x, train_y, test_y = split_data(df, 'y')
    pd.set_option("display.max_columns", 10)
    basic_model_selection(basic_models=None, x=train_x, y=train_y, scoring='accuracy', cv=10)

    # 调参
    # best_est = model_param_tuning_lr(train_x, train_y, test_x, test_y)
    best_est = model_param_tuning_gbdt(train_x, train_y, test_x, test_y)

    pred_label = pd.Series(best_est.predict_proba(test_df)[:, 1], name='pred')
    pred_df = pd.concat([test_id, pred_label], axis=1)
    pred_df.to_csv("result.csv", index=False)

    # 判断模型状态
    plot_learning_curve(best_est, u"学习曲线", train_x, train_y)  # 绘制学习曲线

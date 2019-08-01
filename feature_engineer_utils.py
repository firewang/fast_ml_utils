# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/7/29 9:54
# @Author  : wanghd
# @note    :

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin


class CustomDummifier(TransformerMixin):
    """类别（定类）特征热编码，利用pandas.get_dummies"""

    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, x):
        return pd.get_dummies(x, columns=self.cols)

    def fit(self, *_):
        return self


class CustomEncoder(TransformerMixin):
    """定序特征标签编码（相当于映射为数值, 从小到大）"""

    def __init__(self, col, ordering=None):
        self.ordering = ordering
        self.col = col

    def transform(self, x):
        map_dict = {k: v for k, v in zip(self.ordering, range(len(self.ordering)))}
        # x[self.col] = x[self.col].map(lambda value: self.ordering.index(value))
        x[self.col] = x[self.col].map(map_dict)
        return x

    def fit(self, *_):
        return self


def my_scaler(scaler_obj, training_x, testing_x=None, cols=None, **kwargs):
    """ 标准化和归一化
     scaler_obj : StandardScaler(),MinMaxScaler(),Normalizer()
     training_x 训练集df
     testing_x  测试集df
     cols 需要转换的列list
     **kwargs额外参数 : MinMaxScaler() 缩放范围 feature_range ；Normalizer() 正则化方式 norm """
    scaler_name = scaler_obj.__class__.__name__
    if scaler_name in ["StandardScaler"]:
        pass
    else:
        if kwargs:
            params = scaler_obj.get_params()
            # 更新scaler的参数
            params.update(kwargs)
            scaler_obj.set_params(**params)

    if cols is not None:
        train_x_transformed = pd.DataFrame(scaler_obj.fit_transform(training_x.loc[:, cols]), columns=cols)
        # 删除原始未转换列
        training_x.drop(columns=cols, inplace=True)
        # 构建新的列名index
        new_cols = list(training_x.columns)
        new_cols.extend(cols)
        # 将转换后的列合并回 df
        training_x = pd.merge(training_x, train_x_transformed, left_index=True, right_index=True)
        test_x_transformed = pd.DataFrame(scaler_obj.transform(testing_x.loc[:, cols]), columns=cols)
        # 删除原始未转换列
        testing_x.drop(columns=cols, inplace=True)
        # 将转换后的列合并回 df
        testing_x = pd.merge(testing_x, test_x_transformed, left_index=True, right_index=True)
    else:
        # 如果未传入需要缩放的列， 则对所有列进行作用
        try:
            training_x = pd.DataFrame(scaler_obj.fit_transform(training_x), columns=training_x.columns)
            if testing_x is not None:
                testing_x = pd.DataFrame(scaler_obj.transform(testing_x), columns=testing_x.columns)
        except ValueError:
            print("存在非数值的列")
            raise ValueError
    return scaler_obj, training_x, testing_x


def my_pipeline(names, models):
    my_pipe_line = Pipeline([(name, model) for name, model in zip(names, models)])
    return my_pipe_line


if __name__ == '__main__':
    pass

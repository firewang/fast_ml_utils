# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/7/29 9:54
# @Author  : wanghd
# @note    :

import re
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
        if testing_x is not None:
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


def multiple_replace(text, adict):
    """在多项式生成的结果中，用原始列的名称替换生成列中的相应名称"""
    rx = re.compile('|'.join(map(re.escape, adict)))

    def one_xlat(match):
        return adict[match.group(0)]

    return rx.sub(one_xlat, text)


def my_polynomial(scaler_obj, training_x, testing_x=None, cols=None, **kwargs):
    """ 多项式特征生成PolynomialFeatures
     scaler_obj : 多项式生成器对象
     training_x 训练集df
     testing_x  测试集df
     cols 需要转换的列list
     **kwargs额外参数 : include_bias是否包含偏差列， interaction_only 是否只包含乘积，degree多项式阶数"""
    scaler_name = scaler_obj.__class__.__name__
    if kwargs:
        params = scaler_obj.get_params()
        # 更新scaler的参数
        params.update(kwargs)
        scaler_obj.set_params(**params)

    def genarate_new_column_names(fitted_scaler, orgin_col_names):
        """生成新列名"""
        if fitted_scaler.get_params()["include_bias"]:
            name_dict = {k: v for k, v in zip(fitted_scaler.get_feature_names()[1:11], orgin_col_names)}
        else:
            name_dict = {k: v for k, v in zip(fitted_scaler.get_feature_names()[0:10], orgin_col_names)}
        new_col_names = [multiple_replace(col, name_dict) for col in fitted_scaler.get_feature_names()]
        return new_col_names

    if cols is not None:
        train_x_transformed = pd.DataFrame(scaler_obj.fit_transform(training_x.loc[:, cols]))
        train_x_transformed.columns = genarate_new_column_names(scaler_obj, cols)
        # 删除原始未转换列
        training_x.drop(columns=cols, inplace=True)
        # 将转换后的列合并回 df
        training_x = pd.merge(training_x, train_x_transformed, left_index=True, right_index=True)
        if testing_x is not None:
            test_x_transformed = pd.DataFrame(scaler_obj.transform(testing_x.loc[:, cols]))
            test_x_transformed.columns = genarate_new_column_names(scaler_obj, cols)
            # 删除原始未转换列
            testing_x.drop(columns=cols, inplace=True)
            # 将转换后的列合并回 df
            testing_x = pd.merge(testing_x, test_x_transformed, left_index=True, right_index=True)
    else:
        # 如果未传入需要缩放的列， 则对所有列进行作用
        try:
            origin_cols = training_x.columns
            training_x = pd.DataFrame(scaler_obj.fit_transform(training_x))
            training_x.columns = genarate_new_column_names(scaler_obj, origin_cols)
            if testing_x is not None:
                testing_x = pd.DataFrame(scaler_obj.transform(testing_x))
                testing_x.columns = training_x.columns
        except ValueError:
            print("存在非数值的列")
            raise ValueError
    return scaler_obj, training_x, testing_x


def my_pipeline(names, models):
    my_pipe_line = Pipeline([(name, model) for name, model in zip(names, models)])
    return my_pipe_line


if __name__ == '__main__':
    pass

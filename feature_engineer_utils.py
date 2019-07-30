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


def my_pipeline(names, models):
    my_pipe_line = Pipeline([(name, model) for name, model in zip(names, models)])
    return my_pipe_line


if __name__ == '__main__':
    pass

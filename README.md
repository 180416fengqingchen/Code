#导入计算函数
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from io import StringIO
import pydotplus

#导入训练数据和测试数据
train_data=pd.read_excel('tensile_train.xlsx')
X_train=train_data.iloc[:,2:]
y_train=train_data.iloc[:,1]
test_data=pd.read_excel('tensile_test.xlsx')
X_test=test_data.iloc[:,2:]
y_test=test_data.iloc[:,1]
tree_model = tree.DecisionTreeClassifier()

#网格搜索参数,并打印出参数组合及其分数
param = {'criterion':['gini','entropy'],'max_depth':[1,2,3,4,5,6,7,8,9],'min_samples_leaf':[2,3,5,10],'min_impurity_decrease':[0.1,0.2,0.05]}
grid = GridSearchCV(tree_model,param_grid=param,cv=5)
grid = grid.fit(X_train,y_train)
print('最优分类器:',grid.best_params_,'最优分数:', grid.best_score_)  # 得到最优的参数和分值

#网格搜索到最佳参数组合后，构建最佳参数组合下的决策树
# tree_model = tree.DecisionTreeClassifier(criterion='gini', max_depth=1,  min_samples_leaf=2,min_impurity_decrease=0.1)
# tree_model.fit(X_train, y_train)
# score_train=tree_model.score(X_train,y_train)
# score_test=tree_model.score(X_test,y_test)
# print(score_train)
# print(score_test)


# 画出树形图
# dot_data = StringIO()
# feature_names = ['DBDPE','ZHS','ZS','Sb2O3','MgOH2','DOPO']  #六个特征名称（按数据的顺序）
# target_names = ['Poor', 'Good']  #0类是poor 1类是good
# tree.export_graphviz(tree_model,
#                      out_file=dot_data,
#                      feature_names=feature_names,
#                      class_names=target_names,
#                      filled=True,
#                      rounded=True,
#                      special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("fig_name.pdf")

# Code

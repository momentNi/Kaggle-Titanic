import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
import seaborn as sns

# Load Data
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

# Basic Information
print(data_train.head())
print(data_test.head())
print(data_train.shape, data_test.shape)
print(data_train.info())
print(data_test.info())
print(data_train.describe())
print(data_test.describe())

# Sex Related
color = {0: 'r', 1: 'g'}
sns.countplot(x='Sex', data=data_train, hue='Survived', palette=color)
plt.savefig("不同性别的生存人数.png")
plt.show()

sex_survived_pivot_table = pd.pivot_table(  # 卡方检验验证
    data_train,
    index='Sex',
    columns='Survived',
    values='PassengerId',
    aggfunc='count')
print(sex_survived_pivot_table)
print(chi2_contingency(sex_survived_pivot_table.values)[1])

# Age Related
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(18, 6)
data_train[data_train['Survived'] == 1]['Age'].hist(color='g', ax=axes[0])
axes[0].set_title('Survived Age hist', size=18)
data_train[data_train['Survived'] == 0]['Age'].hist(color='r', ax=axes[1])
axes[1].set_title('Died Age hist', size=18)
plt.savefig("各年龄段的生存人数.png")
plt.show()

# Fare Related
plt.figure(figsize=(18, 8))
sns.distplot(data_train[data_train['Survived'] == 1]['Fare'], color='g')
sns.distplot(data_train[data_train['Survived'] == 0]['Fare'], color='r')
plt.savefig("生存人数与票价之间的关系.png")
plt.show()

# Pclass Related
color = {0: 'r', 1: 'g'}
sns.countplot(x='Pclass', hue='Survived', palette=color, data=data_train)
plt.savefig("生存人数与船舱等级之间的关系.png")
plt.show()

color = {0: 'r', 1: 'g'}
plt.figure(figsize=(18, 6))
sns.boxplot(x='Pclass', y='Fare', data=data_train, hue='Survived', palette=color)
plt.savefig("票价与船舱等级之间的关系.png")
plt.show()

pclass_survived_pivot_table = pd.pivot_table(  # 卡方检验验证
    data_train,
    index='Pclass',
    columns='Survived',
    values=['PassengerId'],
    aggfunc='count')
print(pclass_survived_pivot_table)
print(chi2_contingency(pclass_survived_pivot_table.values)[1])

# Embarked Related
color = {0: 'r', 1: 'g'}
sns.countplot(x='Embarked', hue='Survived', palette=color, data=data_train)
plt.savefig("生存人数与登船港口之间的关系.png")
plt.show()

plt.figure(figsize=(18, 6))
sns.boxplot(
    x='Embarked', y='Fare', data=data_train, hue='Survived', palette=color)
plt.savefig("票价与登船港口之间的关系.png")
plt.show()

embarked_survived_pivot_table = pd.pivot_table(  # 卡方检验验证
    data=data_train,
    index='Embarked',
    columns='Survived',
    values='PassengerId',
    aggfunc='count'
)
print(embarked_survived_pivot_table)
print(chi2_contingency(embarked_survived_pivot_table.values)[1])

# 建立特征工程，主键合并
Y_train = data_train.Survived
PassengerId = data_test.PassengerId
data_train.drop(['Survived'], axis=1, inplace=True)
combined = pd.concat([data_train, data_test], sort=False, axis=0)
combined.drop(['PassengerId'], inplace=True, axis=1)
print(combined.shape)

# 抽取Title特征
combined['title'] = combined['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Don": "Royalty",
    "Rev": "Officer",
    "Dr": "Officer",
    "Mme": "Mrs",
    "Ms": "Mrs",
    "Major": "Officer",
    "Lady": "Royalty",
    "Sir": "Royalty",
    "Mlle": "Miss",
    "Col": "Officer",
    "Capt": "Officer",
    "the Countess": "Royalty",
    "Jonkheer": "Royalty",
    "Dona": 'Mrs'
}
combined['Title'] = combined['title'].map(Title_Dictionary)
print(combined['Title'].value_counts())

del combined['Name']
del combined['title']
print(combined.head())


# 抽取家庭规模
def deal_with_family_size(num):
    if num == 1:
        return 'Singleton'
    elif num <= 4:
        return 'SmallFamily'
    elif num >= 5:
        return 'LargeFamily'
    return num


combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
combined['FamilySize'] = combined['FamilySize'].map(deal_with_family_size)
print(combined['FamilySize'])


# Age缺失值填充，用相同的组别的平均数填充
def select_group_age_median(row):
    condition = ((row['Sex'] == age_group_mean['Sex']) &
                 (row['Pclass'] == age_group_mean['Pclass']) &
                 (row['Title'] == age_group_mean['Title']))
    return age_group_mean[condition]['Age'].values[0]


def age_bin(x):
    if x <= 18:
        return 'child'
    elif x <= 30:
        return 'young'
    elif x <= 55:
        return 'midlife'
    else:
        return 'old'


print(combined.isnull().sum())

age_group_mean = combined.groupby(['Sex', 'Pclass', 'Title'])['Age'].mean().reset_index()
combined['Age'] = combined.apply(lambda x: select_group_age_median(x) if np.isnan(x['Age']) else x['Age'], axis=1)
combined['age_bin'] = combined['Age'].map(age_bin)
print(combined['age_bin'].head())


# Fare缺失值填充，平均数填充
print(combined.isnull().sum())
combined['Fare'].fillna(combined['Fare'].mean(), inplace=True)


# Embarked缺失值填充，众数填充
combined['Embarked'].fillna(combined['Embarked'].mode(), inplace=True)


# Cabin缺失值填充，有值设为yes，无值设为no
combined.loc[combined['Cabin'].notnull(), 'Cabin'] = 'yes'
combined.loc[combined['Cabin'].isnull(), 'Cabin'] = 'no'
print(combined['Cabin'].value_counts())


# 离散型变量处理成独热编码
combined = pd.get_dummies(
    combined,
    columns=['Sex', 'Cabin', 'Pclass', 'Embarked', 'Title', 'FamilySize', 'age_bin'],
    drop_first=True)
combined.drop(['Ticket'], axis=1, inplace=True)
X_train = combined.iloc[:891]
X_test = combined.iloc[891:]
print(X_train.head())
print(X_test.head())


# 模型训练与评估
# 计算不同特征的对结果的重要性
rfc = RandomForestClassifier(n_estimators=100, max_features='sqrt')
scores = cross_val_score(rfc, X_train, Y_train, cv=10)
print(rfc.fit(X_train, Y_train))
print(rfc.feature_importances_)
print(X_train.columns)

feature_importance = pd.Series(rfc.feature_importances_, X_train.columns)
feature_importance.sort_values(ascending=False, inplace=True)
print(feature_importance)

feature_importance.plot(kind='barh')
plt.savefig("不同特征对影响结果的重要性")
plt.show()


# 使用SelectFromModel提供的方法得到特征的重要性
sfm = SelectFromModel(rfc, prefit=True)
train_reduced = sfm.transform(X_train)
print(train_reduced.shape)
print(train_reduced)


# 网格搜索
run_gs = True

if run_gs:
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(
        forest,
        scoring='accuracy',
        param_grid=parameter_grid,
        cv=cross_validation,
        verbose=1)

    grid_search.fit(train_reduced, Y_train)
    # model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else:
    parameters = {
        'bootstrap': False,
        'min_samples_leaf': 1,
        'n_estimators': 10,
        'min_samples_split': 3,
        'max_features': 'log2',
        'max_depth': 8
    }

print(parameters)

model = RandomForestClassifier(**parameters)
model.fit(X_train, Y_train)
y_predict = model.predict(X_test)
res = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_predict})
res.to_csv('预测结果.csv', index=False)


# 学习曲线
def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"Sample Amount")
        plt.ylabel(u"Score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"Train Set Score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"Cross-Validation Score")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.savefig("学习曲线.png")
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


plot_learning_curve(rfc, u"Learning Curve", X_train, Y_train)


# 验证曲线
digits = load_digits()
X = digits.data
Y_train = digits.target
param_range = np.logspace(-6, -1, 5)
vsc = svm.SVC()
train_score, test_score = validation_curve(vsc, X, Y_train,
                                           param_name='gamma',
                                           param_range=param_range,
                                           cv=10,
                                           scoring="accuracy",
                                           n_jobs=1)
train_score_mean = np.mean(train_score, axis=1)
train_score_std = np.std(train_score, axis=1)
test_score_mean = np.mean(test_score, axis=1)
test_score_std = np.std(test_score, axis=1)
plt.title("Validation curve with SVM")
plt.xlabel("Gamma")
plt.ylabel("Score")
plt.ylim()
lw = 2
plt.semilogx(param_range, train_score_mean, label="Training score", color="r", lw=lw)
plt.semilogx(param_range, test_score_mean, label="Cross-validation", color="g", lw=lw)
plt.fill_between(param_range, train_score_mean - train_score_std, train_score_mean + train_score_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig("验证曲线.png")
plt.show()

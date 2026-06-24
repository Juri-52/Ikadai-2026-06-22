import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import setuptools

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# =========================================
# task09: 訓練CSVで学習し、テストCSVで評価しよう
# =========================================

# TODO 1: 学習用CSVとテスト用CSVを読み込む
# - exam_train.csv はモデルを学習するために使う
# - exam_test.csv は最後の確認だけに使う
# - 両方で同じ特徴量を使う
df_train = pd.read_csv("exam_train.csv")
df_test = pd.read_csv("exam_test.csv")
x_train = df_train[["勉強時間","出席率","宿題提出率","小テスト点"]]
y_train = df_train["合格"]
x_test = df_test[["勉強時間","出席率","宿題提出率","小テスト点"]]
y_test = df_test["合格"]

# TODO 2: 2種類の分類モデルを作る
# - DecisionTreeClassifier を学習する
# - LogisticRegression も学習する
# - どちらも学習用CSVだけで fit() する
modelDTC = DecisionTreeClassifier()
modelDTC.fit(x_train, y_train)
modelLR = LogisticRegression()
modelLR.fit(x_train, y_train)

# TODO 3: テスト用CSVで2つのモデルを比べる
# - それぞれのモデルで予測する
# - それぞれの正解率を計算する
# - Decision Tree の混同行列も確認する
predDTC = modelDTC.predict(x_test)
accDTC = accuracy_score(y_test,predDTC)
predLR = modelLR.predict(x_test)
accLR = accuracy_score(y_test,predLR)
matrix = confusion_matrix(y_test,predDTC)

# TODO 4: 予測結果をCSVに保存する
# - テスト用データに2つのモデルの予測結果を追加する
# - task09_predictions.csv として保存する
task09 = pd.DataFrame({
    "勉強時間":df_test["勉強時間"],
    "出席率":df_test["出席率"],
    "宿題提出率":df_test["宿題提出率"],
    "小テスト点":df_test["小テスト点"],
    "正解":y_test,
    "予測1(DTC)":predDTC,
    "予測2(LR)":predLR
})
print(task09)
task09.to_csv("task09_predictions.csv",index=False,encoding="UTF-8")

# TODO 5: 比較結果をグラフで表示する
# - 2つのモデルの正解率を棒グラフにする
# - 正解率と混同行列も print() で表示する
x_plt = ["DecisionTreeClassifier","LogisticRegression"]
y_plt = [accDTC*100,accLR*100]
plt.bar(x_plt,y_plt)
plt.title("モデル別正解率")
plt.xlabel("モデル")
plt.ylabel("正解率")
plt.yticks(range(0,101,10))
plt.show()
print("----------正解率----------")
print("DecisionTreeClassifier:",accDTC*100,"%")
print("LogisticRegression:",accLR*100,"%")
print(matrix)
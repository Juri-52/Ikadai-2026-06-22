import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import setuptools

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =========================================
# task03: Logistic Regression で確率を見よう
# =========================================

# TODO 1: データを準備する
# - students_pass.csv を読み込む
# - 勉強時間、出席率、宿題提出率を特徴量にする
# - 合格/不合格を正解ラベルにする
# - 訓練データとテストデータに分ける
df = pd.read_csv("students_pass.csv")
x = df[["勉強時間","出席率","宿題提出率"]]
y = df["合格"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=7)

# TODO 2: Logistic Regression で分類する
# - LogisticRegression を使って学習する
# - テストデータを予測する
model = LogisticRegression()
model.fit(x_train,y_train)

# TODO 3: 予測の「確率」を確認する
# - predict_proba() を使う
# - 合格する確率だけを取り出してみる
# - 予測クラスと確率を print() で表示する
prob = model.predict_proba(x_test)
print("合格率:",round(prob[0,1],2)*100,"%")
print(prob)

# TODO 4: 確率をグラフで見る
# - テストデータごとの合格確率を棒グラフにする
# - タイトル、軸ラベルをつける
plt.barh(x_test["勉強時間"],prob[:,1])
plt.title("勉強時間別合格率")
plt.xlabel("合格率")
plt.ylabel("勉強時間")
plt.show()
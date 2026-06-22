import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import setuptools

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# =========================================
# task02: テストデータでモデルを確認しよう
# =========================================

# TODO 1: データを準備する
# - students_pass.csv を読み込む
# - 勉強時間、出席率、宿題提出率を特徴量にする
# - 合格/不合格を正解ラベルにする
df = pd.read_csv("students_pass.csv")
x = df[["勉強時間","出席率","宿題提出率"]]
y = df["合格"]

# TODO 2: 訓練データとテストデータに分ける
# - train_test_split を使う
# - テストデータは全体の30%にする
# - random_state を指定して、毎回同じ結果になるようにする
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=7)

# TODO 3: モデルを学習し、テストデータを予測する
# - DecisionTreeClassifier を使う
# - 学習には訓練データだけを使う
# - 予測にはテストデータを使う
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)

# TODO 4: モデルの結果を確認する
# - 正解率を計算する
# - 混同行列を作る
# - 正解と予測を print() で見比べる
acc = accuracy_score(y_test,pred)
matrix = confusion_matrix(y_test,pred)
print("予測データ:",pred)
print("正解データ:",list(y_test))
print(matrix)

# TODO 5: テストデータをグラフで確認する
# - 正解ラベルと予測ラベルの違いが分かるように工夫する
# - タイトル、軸ラベル、凡例をつける
plt.scatter(list(x_test["勉強時間"]),list(y_test),label="正解")
plt.scatter(list(x_test["勉強時間"]),list(pred),label="予測")
plt.title("勉強時間と合否の相関")
plt.xlabel("勉強時間")
plt.ylabel("合否")
plt.legend()
plt.show()

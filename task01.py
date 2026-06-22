import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import setuptools

from sklearn.tree import DecisionTreeClassifier

# =========================================
# task01: CSVを読んで、合格/不合格を予測しよう
# =========================================

# TODO 1: データを読み、分類に使う列を選ぶ
# - students_pass.csv を読み込む
# - 勉強時間と出席率を特徴量にする
# - 合格/不合格の列を正解ラベルにする
df = pd.read_csv("students_pass.csv")
x = df[["勉強時間","出席率"]]
y = df["合格"]

# TODO 2: Decision Tree のモデルを作り、学習させる
# - 授業資料の DecisionTreeClassifier を使う
# - fit() で学習する
model = DecisionTreeClassifier()
model.fit(x,y)

# TODO 3: new_students.csv の学生を予測する
# - 学習で使った特徴量と同じ列を使う
# - 予測結果を print() で確認する
test = pd.read_csv("new_students.csv")
x_test = test[["勉強時間","出席率"]]
pred = model.predict(x_test)
print(pred)

# TODO 4: 結果をグラフで見る
# - 学習データを合格/不合格で色分けする
# - 新しい学生も同じグラフに表示する
# - タイトル、軸ラベル、凡例をつける
train_colors = []
for result in y:
    if result == 1:
        train_colors.append("red")
    elif result == 0:
        train_colors.append("blue")
plt.scatter(df["勉強時間"],df["出席率"],c=train_colors)

test_colors = []
for result in pred:
    if result == 1:
        test_colors.append("red")
    elif result == 0:
        test_colors.append("blue")
plt.scatter(test["勉強時間"],test["出席率"],c=test_colors)

plt.title("勉強時間および出席率と合否の相関")
plt.grid(True)
plt.xlabel("勉強時間")
plt.ylabel("出席率")
plt.scatter([],[],c="red",label="合格")
plt.scatter([],[],c="blue",label="不合格")
plt.legend()
plt.show()
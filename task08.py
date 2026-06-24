import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import setuptools

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# =========================================
# task08: 汚いCSVを整理してから分類しよう
# =========================================

# TODO 1: 汚いCSVを読み、使える形に直す
# - students_dirty.csv を読み込む
# - 「時間」や「%」などの文字を取り除いて数値に変換する
# - 合格/不合格、pass/fail などの表記を 1/0 にそろえる
# - 欠損値がある行をどうするか決める
df = pd.read_csv("students_dirty.csv")
df["勉強時間"] = df["勉強時間"].astype(str).str.replace("時間","",regex=False)
df["勉強時間"] = pd.to_numeric(df["勉強時間"],errors="coerce")
df["出席率"] = df["出席率"].astype(str).str.replace("%","",regex=False)
df["出席率"] = pd.to_numeric(df["出席率"],errors="coerce")
df["宿題提出率"] = df["宿題提出率"].astype(str).str.replace("%","",regex=False)
df["宿題提出率"] = pd.to_numeric(df["宿題提出率"],errors="coerce")
df["結果"] = df["結果"].astype(str).str.upper()
df.loc[
    (df["結果"] == "FAIL") | (df["結果"] == "不合格"),"結果"
] = "0"
df.loc[
    (df["結果"] == "PASS") | (df["結果"] == "合格"),"結果"
] = "1"
df.to_csv("cleaned_data.csv",index=False,encoding="UTF-8")

# TODO 2: 整理したデータで分類モデルを作る
# - 勉強時間、出席率、宿題提出率を特徴量にする
# - 合格/不合格を正解ラベルにする
# - 訓練データとテストデータに分けて学習する
x = df[["勉強時間","出席率","宿題提出率"]]
y = df["結果"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=7)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)

# TODO 3: テスト結果を確認して保存する
# - テストデータを予測する
# - 正解率を計算して表示する
# - テストデータ、正解、予測をまとめて task08_predictions.csv に保存する
pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
print(x_test)
print(y_test)
print(pred)
print("正解率:",acc*100,"%")
task08 = pd.DataFrame({
    "勉強時間":x_test["勉強時間"],
    "出席率":x_test["出席率"],
    "宿題提出率":x_test["宿題提出率"],
    "正解":y_test,
    "予測":pred
})
print(task08)
task08.to_csv("task08_predictions.csv",index=False,encoding="UTF-8")

# TODO 4: 整理後のデータをグラフで見る
# - 合格/不合格で色分けした散布図を作る
# - タイトル、軸ラベルをつける
colors = []
for i in y:
    if i == "1":
        colors.append("red")
    else:
        colors.append("blue")
plt.scatter(df["出席率"],df["宿題提出率"],c=colors)
plt.scatter([],[],c="red",label="合格")
plt.scatter([],[],c="blue",label="不合格")
plt.title("合否の傾向")
plt.xlabel("出席率")
plt.ylabel("宿題提出率")
plt.xticks(range(0,101,10))
plt.yticks(range(0,101,10))
plt.legend()
plt.grid(True)
plt.show()
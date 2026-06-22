import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import setuptools

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# =========================================
# task04: メールを迷惑メール/通常メールに分類しよう
# =========================================

# TODO 1: メールデータを準備する
# - emails.json を読み込む
# - 単語数、リンク数、緊急ワード数、添付ファイルの有無を特徴量にする
# - 迷惑メールかどうかを正解ラベルにする
df = pd.read_json("emails.json")
x = df[["単語数","リンク数","緊急ワード数","添付ファイル有無"]]
y = df["迷惑メール"]

# TODO 2: モデルを作って評価する
# - データを訓練用とテスト用に分ける
# - DecisionTreeClassifier で学習する
# - テストデータを予測する
# - 正解率と混同行列を確認する
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=7)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
matrix = confusion_matrix(y_test,pred)
print("正解率:",acc*100,"%")

# TODO 3: 間違い方を考える
# - 混同行列を見て、どんな間違いがあるか print() で確認する
# - 通常メールを迷惑メールにする間違いと、迷惑メールを見逃す間違いを意識する
print(matrix)

# TODO 4: グラフで特徴量を見る
# - リンク数と緊急ワード数など、分かりやすい2列を選ぶ
# - 迷惑メール/通常メールで色分けする
# - タイトルと軸ラベルをつける
colors = []
for result in df["迷惑メール"]:
    if result == 1:
        colors.append("red")
    else:
        colors.append("blue")
plt.scatter(df["リンク数"],df["緊急ワード数"],c=colors)
plt.scatter([],[],c="red",label="迷惑メール")
plt.scatter([],[],c="blue",label="通常メール")
plt.title("メール分類")
plt.xlabel("リンク数")
plt.ylabel("緊急ワード数")
plt.grid(True)
plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import setuptools

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# =========================================
# task06: 3つのカテゴリに分類しよう
# =========================================

# TODO 1: 問い合わせデータを分類用に準備する
# - support_tickets.json を読み込む
# - 文章の長さ、料金ワード数、技術ワード数、契約ワード数を特徴量にする
# - 問い合わせカテゴリを正解ラベルにする
df = pd.read_json("support_tickets.json")
x = df[["文章長","料金ワード数","技術ワード数","契約ワード数"]]
y = df["カテゴリ"]

# TODO 2: 3クラス分類モデルを作る
# - 訓練データとテストデータに分ける
# - DecisionTreeClassifier で学習する
# - テストデータのカテゴリを予測する
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=9)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)

# TODO 3: 結果を確認する
# - 正解率を計算する
# - 混同行列で、どのカテゴリを間違えたかを見る
# - 予測結果、正解率、混同行列を print() で表示する
acc = accuracy_score(y_test,pred)
matrix = confusion_matrix(y_test,pred)
print("予測:",pred)
print("正解率:",acc*100,"%")
print(matrix)

# TODO 4: 混同行列を画像として表示する
# - plt.imshow() を使う
# - タイトルとカラーバーをつける
# - 余裕があれば、軸にカテゴリ名を表示する
plt.imshow(matrix,interpolation='nearest',cmap=plt.cm.Blues)
plt.title("カテゴリ")
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.xticks(range(3),labels=["契約","技術","料金"])
plt.yticks(range(3),labels=["契約","技術","料金"])
plt.colorbar()
plt.show()
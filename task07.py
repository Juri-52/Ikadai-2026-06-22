import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import setuptools

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# =========================================
# task07: 特徴量重要度を見よう
# =========================================

# TODO 1: アプリ利用データを準備する
# - app_users.csv を読み込む
# - ログイン日数、学習時間、完了レッスン数、質問数を特徴量にする
# - 継続したかどうかを正解ラベルにする
df = pd.read_csv("app_users.csv")
x = df[["ログイン日数","学習時間","完了レッスン数","質問数"]]
y = df["継続"]

# TODO 2: Decision Tree を学習する
# - 訓練データとテストデータに分ける
# - 木の深さを制限して学習する
# - テストデータも予測してみる
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
model = DecisionTreeClassifier(max_depth=3)
model.fit(x_train,y_train)
pred = model.predict(x_test)

# TODO 3: どの特徴量が大事だったかを見る
# - 学習したモデルから feature_importances_ を取り出す
# - 特徴量名と重要度を print() で表示する
fi = model.feature_importances_
for a,b in zip(x,fi):
    print(a,":",b*100,"%")

# TODO 4: 特徴量重要度をグラフにする
# - 棒グラフで表示する
# - タイトル、軸ラベルをつける
x = x.columns.tolist()
fi = fi*100
plt.bar(x,fi)
plt.title("特徴量重要度")
plt.xlabel("特徴量")
plt.ylabel("重要度")
plt.yticks(range(0,101,10))
plt.show()

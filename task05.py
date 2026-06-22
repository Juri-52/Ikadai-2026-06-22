import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import setuptools

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# =========================================
# task05: max_depth を変えて過学習を確認しよう
# =========================================

# TODO 1: 健康診断データを準備する
# - health_check.csv を読み込む
# - 年齢、BMI、血圧、血糖値、運動日数を特徴量にする
# - risk を正解ラベルにする
# - 訓練データとテストデータに分ける
df = pd.read_csv("health_check.csv")
x = df[["年齢","BMI","血圧","血糖値","運動日数"]]
y = df["リスク"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# TODO 2: max_depth を変えて何度か学習する
# - 深さ 1, 2, 3, 制限なしを試す
# - それぞれのモデルで訓練データとテストデータの正解率を出す
# 制限なし
model0 = DecisionTreeClassifier()
model0.fit(x_train,y_train)

# 深さ1
model1 = DecisionTreeClassifier(max_depth=1)
model1.fit(x_train,y_train)

# 深さ2
model2 = DecisionTreeClassifier(max_depth=2)
model2.fit(x_train,y_train)

# 深さ3
model3 = DecisionTreeClassifier(max_depth=3)
model3.fit(x_train,y_train)

# TODO 3: 過学習していないか確認する
# - 訓練データだけ高く、テストデータが低い場合は過学習の可能性がある
# - 結果を print() で表示する
train = []
test = []

pred_train = model0.predict(x_train)
acc_train = accuracy_score(y_train,pred_train)
pred_test = model0.predict(x_test)
acc_test = accuracy_score(y_test,pred_test)
print("無制限/訓練データ:",acc_train*100,"%　テストデータ:",acc_test*100,"%")
train.append(acc_train)
test.append(acc_test)

pred_train = model1.predict(x_train)
acc_train = accuracy_score(y_train,pred_train)
pred_test = model1.predict(x_test)
acc_test = accuracy_score(y_test,pred_test)
print("深さ１/訓練データ:",acc_train*100,"%　テストデータ:",acc_test*100,"%")
train.append(acc_train)
test.append(acc_test)

pred_train = model2.predict(x_train)
acc_train = accuracy_score(y_train,pred_train)
pred_test = model2.predict(x_test)
acc_test = accuracy_score(y_test,pred_test)
print("深さ２/訓練データ:",acc_train*100,"%　テストデータ:",acc_test*100,"%")
train.append(acc_train)
test.append(acc_test)

pred_train = model3.predict(x_train)
acc_train = accuracy_score(y_train,pred_train)
pred_test = model3.predict(x_test)
acc_test = accuracy_score(y_test,pred_test)
print("深さ３/訓練データ:",acc_train*100,"%　テストデータ:",acc_test*100,"%")
train.append(acc_train)
test.append(acc_test)

# TODO 4: 正解率の変化をグラフにする
# - 深さごとの訓練スコアとテストスコアを折れ線グラフで表示する
# - タイトル、軸ラベル、凡例をつける
plt.plot(train,marker="o",label="訓練データ")
plt.plot(test,marker="o",label="テストデータ")
plt.xlabel("深さ")
plt.ylabel("正解率")
plt.legend()
plt.show()
※作成中

# EDA(Exploratory Data Analysis)

- [EDA(Exploratory Data Analysis)](#edaexploratory-data-analysis)
  - [目的](#目的)
  - [まず初めに](#まず初めに)
  - [初めのうちに使うと良いEDAライブラリ](#初めのうちに使うと良いedaライブラリ)
  - [データの作成された背景を考えてみる](#データの作成された背景を考えてみる)
  - [データ加工](#データ加工)
    - [フィルタリング](#フィルタリング)
    - [ソート](#ソート)
  - [データの分布を知ろう](#データの分布を知ろう)
    - [value_counts](#value_counts)
    - [ヒストグラム](#ヒストグラム)
    - [Countplot](#countplot)
    - [散布図](#散布図)
    - [ベン図](#ベン図)
  - [時系列](#時系列)
    - [折れ線グラフ](#折れ線グラフ)
    - [statsmodels](#statsmodels)
  - [最後に](#最後に)

## 目的

- 特徴量や目的変数の関係/相関を分析する
- データに隠された傾向をあばき、新しい特徴量につながる気付きを得る

## まず初めに

行列の確認`train_df.shape`  
最初の5行`train_df.head()`  
最後の5行`train_df.tail()`  
欠損値の確認`train_df.isnull().sum()`  
全てnsnの列削除`train_df.dropna(axis=1, how='all')`  
値はあるけど、全て同じ値の列削除  

```python
for col in train_df.columns:
    value_counts = train_df[col].nunique()
    if value_counts == 1:
        train_df = train_df.drop([col], axis=1)
```

データ自体の情報`train_df.info()`  
各カラムの統計`train_df.describe()`  
  groupby()を使った統計`train_df.groupby(["category"]).describe()`  
各カラムごとの相関  

```python
## ヒートマップで表示
!pip install sweetviz

import sweetviz

train_df.corr()
## 表示サイズ設定
fig, ax = plt.subplots(figsize=(12, 9)) 
sns.heatmap(df, square=True, vmax=1, vmin=-1, center=0)
```

この段階で一度ベースラインを作成するのお勧め  
`LightGBM`を使用して`model.feature_importances_`で重要度の高い変数を把握して、それらからEDAをするのがコスパ良いと思う

## 初めのうちに使うと良いEDAライブラリ

sweetviz  
各カラムの分布、相関がhtml形式で一覧できる

```python
train_df = pd.read_csv(INPUT_PATH + 'train.csv')
test_df = pd.read_csv(INPUT_PATH + 'test.csv')

my_report = sv.compare([train_df, 'Train'], [test_df, 'Test'], 'target')
my_report.show_html(OUTPUT_PATH + 'sweetsviz_report.html')
```

## データの作成された背景を考えてみる

前提条件：重要度の高い変数、相関の高い変数を把握済み  

- なんでこのデータが重要なのか
  - 人間が予測してた時もそのデータを使っていたか
  - 単純に相関が高いのか、それとも一部のデータが非常に重要な役割を果たしているのか

## データ加工

### フィルタリング

気になった分布、期間でフィルタリングしてみる  

```python
train_df[train_df['column'] == 'value']
```

### ソート

```python
train_df.sort_values('column', axis=1, ascending=False)
```

## データの分布を知ろう

### value_counts

どんなデータが格納されているか確認する時に便利

```python
train_df['column'].value_counts()
```

### ヒストグラム  

連続変数の分布を知るために必要  
外れ値が知れるかも  
周期的にデータが多い分布の列があるかも

```python
## bin:棒の数
train_df.select_dtypes(np.number).hist(bins = 50,figsize =(30,20),color='orange')
```

### Countplot

カテゴリ変数の分布を知るために必要  
大分偏りがあるかも  

```python
## 1カテゴリ変数で描画
sns.countplot(x='category1', data=train_df)

## 2カテゴリ変数で描画
sns.countplot(x='category1', hue='category2', data=train_df)
```

### 散布図  

複数カラムの分布を知るために必要  
カテゴリごとの分布が知れるかも  
外れ値の影響でグラフが見づらかったりするかも  
緯度経度を散布図で見てみると、多い分布(市区町村)が分かるかも

```python
plt.scatter(train_df['column'], train_df['column'])
```

### ベン図

複数カラムのカテゴリ変数の分布を知るために必要  
学習データとテストデータでどれくらい重なりがあるのか  
まったく重なっていないなら、その列は削除した方が良い

```python
from matplotlib_venn import venn2

venn2([set(train_df['category']), set(test['category'])], set_labels=('train', 'test'))
```

## 時系列

### 折れ線グラフ

そのまま相関があるのか、1日ずれて相関がありそうか(ラグ特徴量作成につながる)  
月ごと、週ごとにgroupby()して見てみるのもあり

```python
train_df['date'] = pd.to_datetime(train_df[['year', 'month', 'day']])

plt.plot(train_df['date'], train_df[['column']], color='blue')
plt.plot(train_df['date'], train_df[['column']], color='black')
```

### statsmodels

トレンド、季節、残差に分解できるライブラリ

```python
## 時系列分解して可視化
!pip install statsmodels

from statsmodels.tsa.seasonal import STL 

#STL分解
## period:周期性
stl=STL(train_df['column'], period=12, robust=True)
stl_series = stl.fit()

## STL分解結果のグラフ化
stl_series.plot()
plt.show()
```

```python
## 時系列分解して特徴量追加できる
import statsmodels.api as sm

result = sm.tsa.seasonal_decompose(train_df['column'], period=30)

## トレンド
train_df['column_trend'] = result.trend    
## 残差
train_df['column_reside'] = result.reside   
## 季節性データ
train_df['column_seasonal'] = result.seasonal 
```

## 最後に

ライブラリ等で簡単に実装できる特徴量エンジニアリングは、正直知っていればみんな実装できます。  
しかし、ドメイン知識を得てEDAをして得たアイデアは、もしかしたら誰も考えつかなかった観点かもしれません。  
そういった点でEDAは非常に重要なので、定期的に実施することをお勧めします。

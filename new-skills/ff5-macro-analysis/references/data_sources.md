# データソース詳細・引用形式

## 1. Fama-French 5ファクター

### 提供元
Kenneth R. French - Tuck School of Business, Dartmouth University

### アクセス方法
- **公式ページ**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **直接ダウンロードURL**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip
- **更新頻度**: 月次（毎月更新）
- **開始時点**: 1963年7月
- **ファイル形式**: CSV（ZIP圧縮）

### 含まれる変数
| 変数 | 説明 | 単位 |
|------|------|------|
| Mkt-RF | 市場超過リターン（市場ポートフォリオ - 無リスク金利） | % |
| SMB | Small Minus Big（小型株 - 大型株） | % |
| HML | High Minus Low（高BPR - 低BPR） | % |
| RMW | Robust Minus Weak（高収益性 - 低収益性） | % |
| CMA | Conservative Minus Aggressive（保守的投資 - 積極的投資） | % |
| RF | 無リスク金利（1ヶ月T-Bill利回り） | % |

### 引用形式（APA）
```
French, K. R. (2024). Fama/French 5 Factors (2x3) [Data file].
  Retrieved from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
```

### 引用形式（論文本文）
```
Fama-French 5ファクターの月次データは、Kenneth French's Data Library（https://mba.tuck.dartmouth.edu/
pages/faculty/ken.french/data_library.html）より取得した（取得日：[取得日を記入]）。
```

---

## 2. FRED（Federal Reserve Economic Data）

### 提供元
Federal Reserve Bank of St. Louis

### アクセス方法
- **公式ページ**: https://fred.stlouisfed.org/
- **APIドキュメント**: https://fred.stlouisfed.org/docs/api/fred/
- **APIキー取得（無料）**: https://fred.stlouisfed.org/docs/api/api_key.html
- **更新頻度**: 系列により異なる（下記参照）

### 使用系列一覧

| 変数名（本研究） | FRED系列ID | 説明 | 周期 | 開始時点 |
|----------------|-----------|------|------|---------|
| CPI_growth | CPIAUCSL | 消費者物価指数（全都市、全品目） | 月次 | 1947年1月 |
| IP_growth | INDPRO | 鉱工業生産指数 | 月次 | 1919年1月 |
| TERM_SPREAD | GS10 - TB3MS | 10年国債利回り - 3ヶ月T-Bill | 月次 | 1953年4月 |
| DEF_SPREAD | DBAA - DAAA | BAA格社債 - AAA格社債（Moody's） | 月次 | 1919年1月 |
| FX_return | DTWEXBGS | 広義ドル指数（名目、日次→月次末値） | 日次 | 2006年1月 |
| OIL_return | DCOILWTICO | WTI原油スポット価格（日次→月次末値） | 日次 | 1986年1月 |
| VIX | VIXCLS | CBOE Volatility Index（日次→月次平均） | 日次 | 1990年1月 |

### 各系列へのダイレクトリンク
- CPI: https://fred.stlouisfed.org/series/CPIAUCSL
- 鉱工業生産: https://fred.stlouisfed.org/series/INDPRO
- 10年国債: https://fred.stlouisfed.org/series/GS10
- 3ヶ月T-Bill: https://fred.stlouisfed.org/series/TB3MS
- BAA社債: https://fred.stlouisfed.org/series/DBAA
- AAA社債: https://fred.stlouisfed.org/series/DAAA
- ドル指数: https://fred.stlouisfed.org/series/DTWEXBGS
- WTI原油: https://fred.stlouisfed.org/series/DCOILWTICO
- VIX: https://fred.stlouisfed.org/series/VIXCLS

### 引用形式（APA）
```
Federal Reserve Bank of St. Louis. (2024). FRED Economic Data [Data file].
  Retrieved from https://fred.stlouisfed.org/
```

### 個別系列の引用形式（例：CPI）
```
Federal Reserve Bank of St. Louis. (2024). Consumer Price Index for All Urban Consumers:
  All Items in U.S. City Average [CPIAUCSL]. FRED, Federal Reserve Bank of St. Louis.
  Retrieved from https://fred.stlouisfed.org/series/CPIAUCSL
```

---

## 3. データに関する重要な注意事項

### 遡及改定
FREDのデータは遡及改定されることがあります。論文に使用するデータの取得日を必ず記録し、
`01_data_collection.py` が生成する `metadata.json` に保存してください。

### VIXのサンプル期間制限
VIXCLSは1990年1月から利用可能です。VIXを含む分析はサブサンプル期間（1990年〜）に限定してください。
本論文ではメインサンプル期間（2006〜2025年）は全期間VIXが利用可能なため影響はありませんが、
より長いサンプル期間を検討する際は注意が必要です。

### 為替データの選択について
DTWEXBGS（広義ドル指数）は26か国以上の通貨を対象とした総合指数で、
FREDが提供する最も包括的なドル指数です。2006年以降のデータが利用可能です。

### Moody's社債利回りについて
DBAA（BAA格）とDAAA（AAA格）はMoody's社が算出しており、FREDを通じて公開されています。
デフォルトスプレッド = DBAA - DAAA として計算します。

---

## 4. 再現性の確保

### ランダムシードの固定
```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
rf = RandomForestRegressor(random_state=RANDOM_STATE, ...)
```

### 取得日時の記録
スクリプト実行時に `data_metadata.json` に取得日時が自動記録されます。
論文本文または付録に取得日時を明記してください。

---

## 5. 倫理・利用規約

- Kenneth French Data Library: 学術研究目的での使用は明示的に許可されています
- FRED: 連邦政府機関のデータとして、学術・商用問わず無料で利用可能です
  （ただし出典の明記が推奨されます）

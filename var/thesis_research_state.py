from pathlib import Path


MASTER_PLAN_PATH = Path(
    "/Users/mitsuyoshitsuha/Documents/研究論文/修士論文研究計画書：Fama-Frenchファクターとマクロ経済変数の相互作用、および個別株予測における因果性の解明.pdf"
)

RESEARCH_DIR = Path("/Users/mitsuyoshitsuha/Documents/研究論文")
PPT_OUTPUT_PATH = RESEARCH_DIR / "修士論文_執筆整理_進捗資料.pptx"
XLSX_OUTPUT_PATH = RESEARCH_DIR / "修士論文_研究整理ベース.xlsx"
CODE_OUTPUT_PATH = RESEARCH_DIR / "修士論文_進捗管理コード.py"
DOCX_OUTPUT_PATH = RESEARCH_DIR / "修士論文_統合進捗レポート.docx"

THESIS_TITLE = "Fama-Frenchファクターとマクロ経済変数の相互作用、および個別株予測における因果性の解明"
SHORT_TITLE = "修士論文の現在地"

SUMMARY = [
    "最終計画書を主データとして、他資料は参考情報として扱う。",
    "機械学習の高い個別株予測力の源泉を、Fama-Frenchファクターとマクロ経済変数の因果特性から解明する。",
    "主軸は『予測モデルを増やすこと』ではなく、『どのファクターが経済に対して先行・遅行するかを構造的に示すこと』に置く。",
]

BACKGROUND = [
    "個別株予測では、ニューラルネットワークや決定木が高い予測精度を示す先行研究がある。",
    "一方で、MLモデルはブラックボックス性が高く、何が予測を可能にしているかの経済的説明が弱い。",
    "本研究は、計量経済学的な因果性分析を使い、その予測力の源泉を解明しようとする。",
]

PURPOSES = [
    "Fama-Frenchファクターと主要マクロ変数の間に、どのような先行・遅行関係があるかを明らかにする。",
    "MLモデルで観測される高い決定係数が、特定ファクターの遅行特性に由来するかを検証する。",
    "産業別・財務別などの属性を持つファクターが、経済に対してどのような時間差を示すかを調べる。",
    "ブロック外生性VARを拡張し、金融市場とマクロ経済の双方向構造を包括的に分析する。",
]

HYPOTHESES = [
    {
        "id": "H1",
        "title": "情報効率性と双方向因果性",
        "detail": "金融市場（FFファクター）はマクロ経済を先取りする一方、マクロ経済もFFファクターに先行する。両者の間には双方向の先行遅行構造がある。",
    },
    {
        "id": "H2",
        "title": "遅行ファクターの予測力",
        "detail": "MLで観測される高い個別株予測力の一部は、経済に対して遅行する特定のFFファクター、特に産業別・財務別ファクターの存在で説明できる。",
    },
    {
        "id": "H3",
        "title": "CPI→RMW の頑健な先行性",
        "detail": "CPI は、多重検定補正後も頑健に RMW に対して統計的に有意な先行性を持つ。",
    },
]

THREE_LAYER_STRUCTURE = [
    {
        "layer": "先行層",
        "variables": "CPI、為替、原油",
        "role": "将来の経済活動を先読みする層",
        "interpretation": "インフレ期待、国際経済状況、資源価格の変動が、金融市場や実体経済に先行して影響する。",
    },
    {
        "layer": "同時・双方向層",
        "variables": "FFファクター（Mkt-RF, SMB, HML, RMW, CMA）",
        "role": "金融市場の動向を表す層",
        "interpretation": "マクロ経済の期待を織り込みつつ、実体経済・マクロ金融変数にも影響を与える。",
    },
    {
        "layer": "遅行層",
        "variables": "IP、タームスプレッド、デフォルトスプレッド",
        "role": "経済活動の結果として現れる層",
        "interpretation": "先行層や金融市場の動きに遅れて反応する。",
    },
]

DATA_PLAN = [
    "Fama-French 5ファクターを基本に、必要なら産業別・財務別ファクターも追加検討する。",
    "マクロ変数は IP, CPI, タームスプレッド, デフォルトスプレッド, 為替, 原油, VIX を中心に扱う。",
    "現在のデータ期間は 2006年以降・239か月と短いため、リーマンショック以前を含む長期月次データの取得と再走を最優先にする。",
]

METHODS = [
    "3層構造のブロック外生性VARを再構築する。",
    "BIC でラグ次数を選びつつ、CPI→RMW では 1, 3, 6, 12か月など複数ラグも検証する。",
    "グレンジャー因果性を各変数ペア・ブロック間で検定し、逆方向の因果性も確認する。",
    "Bonferroni などの多重検定補正を適用して、頑健な発見を強調する。",
    "IRF と FEVD で動的伝播と相対的重要性を可視化する。",
    "OOS R² と方向的中率で、特に CPI→RMW 関係の予測的有用性を評価する。",
    "非定常変数は差分化し、VIX は対数差分または標準化でスケール調整する。",
    "VAR残差に t分布を適用し、ファットテールを考慮したリスク評価を行う。",
]

CONTRIBUTIONS = [
    "MLのブラックボックス問題に対し、予測力の源泉をグレンジャー因果性で解明しようとする点。",
    "FFファクターとマクロ経済の双方向の先行遅行構造を、時系列的に明らかにする点。",
    "CPI→RMW の頑健な先行関係を、時系列の因果性として補強する点。",
    "ブロック外生性VARを FFファクター予測へ応用し、有効性と限界を検証する点。",
    "遅行ファクターの発見を通じて、ポートフォリオ戦略やリスク管理への実務的示唆を与える点。",
]

ROADMAP = [
    "正規データの取得と再走を最優先で進める。",
    "非定常変数の差分化、VIX のスケール調整など前処理を行う。",
    "3層構造のブロック外生性VARを再構築し、ラグを選択する。",
    "グレンジャー因果性検定と多重検定補正を実行する。",
    "IRF と FEVD で動的分析を行う。",
    "OOS R² と方向的中率を評価する。",
    "結果を解釈し、先行研究と比較しながら考察を書く。",
    "論文全体を執筆・推敲する。",
]

CURRENT_PROGRESS = [
    "最終計画書ベースの研究軸は確定している。",
    "研究の中核は『MLの予測力の源泉解明』と『FFファクターとマクロの因果性分析』で明確。",
    "当初の4層構造は棄却され、現在は 3層構造で再構築する方針になっている。",
    "CPI→RMW を重点仮説として扱う方向が明示されている。",
    "最大の未了項目は、長期の正規データ取得と再走。",
]

CURRENT_ISSUES = [
    "現行サンプルは 2006年以降 239か月で短く、検出力に限界がある。",
    "ブロック階層は既存案から修正されており、資料間の古い表現を整理し直す必要がある。",
    "ML論文の数値は厳密に扱う必要があり、OOS R² を『40%のリスク』のように言い換えないことが重要。",
]

CHAPTER_STATUS = [
    ("第I章", "はじめに", "草稿あり", "最終計画書の目的・背景に合わせて文言を揃える段階。"),
    ("第II章", "先行研究と理論的背景", "草稿あり", "MLブラックボックス問題と FF-マクロ相互作用の橋渡しを補強したい。"),
    ("第III章", "モデル構築", "設計済み", "3層構造、ラグ選択、差分化、t分布の位置づけを明文化する段階。"),
    ("第IV章", "実証分析", "設計済み", "因果性検定・IRF/FEVD・OOS 評価の順番は明確。"),
    ("第V章", "結果と考察", "未着手", "結果が揃い次第、仮説ごとに整理する。"),
    ("第VI章", "結論", "設計済み", "双方向因果性、遅行ファクター、CPI→RMW を軸にまとめる方針。"),
]

REFERENCE_FILES = [
    {
        "id": "M01",
        "name": "修士論文研究計画書：Fama-Frenchファクターとマクロ経済変数の相互作用、および個別株予測における因果性の解明.pdf",
        "role": "主データ・最終計画書",
        "priority": "最高",
        "note": "PowerPoint と Excel はこの文書を最優先で反映する。",
    },
    {
        "id": "R01",
        "name": "Empirical Asset Pricing via Machine Learning.pdf",
        "role": "ML予測力の背景説明",
        "priority": "高",
        "note": "OOS R² 0.33-0.40、重要シグナル、ブラックボックス問題の補足に使う。",
    },
    {
        "id": "R02",
        "name": "第Ⅱ章_先行研究と理論的背景.docx",
        "role": "補助原稿",
        "priority": "中",
        "note": "最終計画書に沿うように再編集する前提で参照する。",
    },
    {
        "id": "R03",
        "name": "修士論文_研究計画_最終まとめ.docx",
        "role": "過去の統合メモ",
        "priority": "中",
        "note": "重複やズレの確認用。主データではない。",
    },
]

WEB_SOURCES = [
    {
        "category": "Academic paper",
        "name": "Empirical Asset Pricing via Machine Learning",
        "url": "https://academic.oup.com/rfs/article-abstract/33/5/2223/5758276",
        "note": "Gu, Kelly, Xiu (2020). Trees and neural networks perform best; dominant signals include momentum, liquidity, and volatility.",
    },
    {
        "category": "Academic paper",
        "name": "A Comprehensive Look at The Empirical Performance of Equity Premium Prediction",
        "url": "https://academic.oup.com/rfs/article/21/4/1455/1565737",
        "note": "Welch and Goyal (2008). Most proposed equity-premium predictors perform poorly out of sample.",
    },
    {
        "category": "Academic paper",
        "name": "The Fama-French’s five-factor model relation with interest rates and macro variables",
        "url": "https://www.sciencedirect.com/science/article/pii/S1062940820300942",
        "note": "De Oliveira et al. (2020). CPI innovations may proxy for RMW; useful for the CPI→RMW hypothesis.",
    },
    {
        "category": "Data source",
        "name": "Kenneth French Data Library: Fama/French 5 Factors (2x3)",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html",
        "note": "Official construction notes and downloadable monthly factor returns.",
    },
    {
        "category": "Data source",
        "name": "FRED INDPRO",
        "url": "https://fred.stlouisfed.org/series/INDPRO",
        "note": "Industrial Production: Total Index, monthly, seasonally adjusted.",
    },
    {
        "category": "Data source",
        "name": "FRED CPIAUCSL",
        "url": "https://fred.stlouisfed.org/series/CPIAUCSL",
        "note": "CPI for All Urban Consumers, monthly, seasonally adjusted.",
    },
    {
        "category": "Data source",
        "name": "FRED T10Y3M",
        "url": "https://fred.stlouisfed.org/series/T10Y3M",
        "note": "10-Year Treasury minus 3-Month Treasury, daily with monthly format available.",
    },
    {
        "category": "Data source",
        "name": "FRED BAA10YM",
        "url": "https://fred.stlouisfed.org/series/BAA10YM",
        "note": "Monthly Baa corporate spread relative to 10-year Treasury.",
    },
    {
        "category": "Data source",
        "name": "FRED DTWEXBGS",
        "url": "https://fred.stlouisfed.org/series/DTWEXBGS",
        "note": "Nominal Broad U.S. Dollar Index, daily with monthly format available.",
    },
    {
        "category": "Data source",
        "name": "FRED DCOILWTICO",
        "url": "https://fred.stlouisfed.org/series/DCOILWTICO",
        "note": "WTI crude oil spot price, daily with monthly format available.",
    },
    {
        "category": "Data source",
        "name": "FRED VIXCLS",
        "url": "https://fred.stlouisfed.org/series/VIXCLS",
        "note": "CBOE Volatility Index, daily close series.",
    },
]

VARIABLE_CATALOG = [
    ("Mkt-RF", "FF factor", "同時・双方向層", "Kenneth French", "Monthly", "原系列", "市場リスクプレミアム"),
    ("SMB", "FF factor", "同時・双方向層", "Kenneth French", "Monthly", "原系列", "サイズ要因"),
    ("HML", "FF factor", "同時・双方向層", "Kenneth French", "Monthly", "原系列", "簿価対市場価値要因"),
    ("RMW", "FF factor", "同時・双方向層", "Kenneth French", "Monthly", "原系列", "堅実な収益性要因。CPI→RMW を重点確認"),
    ("CMA", "FF factor", "同時・双方向層", "Kenneth French", "Monthly", "原系列", "投資要因"),
    ("INDPRO", "Macro", "遅行層", "FRED", "Monthly", "差分または成長率", "鉱工業生産"),
    ("CPIAUCSL", "Macro", "先行層", "FRED", "Monthly", "差分またはインフレ率", "消費者物価指数"),
    ("T10Y3M", "Macro", "遅行層", "FRED", "Daily/Monthly", "差分も検討", "タームスプレッド"),
    ("BAA10YM", "Macro", "遅行層", "FRED", "Monthly", "差分も検討", "デフォルトスプレッド"),
    ("DTWEXBGS", "Macro", "先行層", "FRED", "Daily/Monthly", "対数差分候補", "為替インデックス"),
    ("DCOILWTICO", "Macro", "先行層", "FRED", "Daily/Monthly", "対数差分候補", "WTI原油価格"),
    ("VIXCLS", "Market stress", "市場ストレス補助", "FRED", "Daily/Monthly", "対数差分または標準化", "VIX"),
]


def exportable_snapshot():
    return {
        "master_plan_path": str(MASTER_PLAN_PATH),
        "thesis_title": THESIS_TITLE,
        "summary": SUMMARY,
        "background": BACKGROUND,
        "purposes": PURPOSES,
        "hypotheses": HYPOTHESES,
        "three_layer_structure": THREE_LAYER_STRUCTURE,
        "data_plan": DATA_PLAN,
        "methods": METHODS,
        "contributions": CONTRIBUTIONS,
        "roadmap": ROADMAP,
        "current_progress": CURRENT_PROGRESS,
        "current_issues": CURRENT_ISSUES,
        "chapter_status": CHAPTER_STATUS,
        "reference_files": REFERENCE_FILES,
    }

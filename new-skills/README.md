# Claude 向けスキルパッケージ（修正反映版）

元の `skill` リポジトリ（`/Users/mitsuyoshitsuha/skill` の `2/`・`skill/`）をベースにしたうえで、**`new/` 内の ZIP**（動作確認済みのスクリプト・テンプレート入り）を展開し、実行可能な形で揃えたものである。

- **アーカイブの場所**: `new/*.zip`（再配布やバックアップ用にそのまま残している）
- **展開先**: 本ディレクトリ直下の各 `*/SKILL.md` と同階層の `scripts/`・`skills/` 等

## 同梱スキルとコード

| フォルダ | 内容 |
|----------|------|
| `quant-deep-research/` | 定量調査4フェーズ・レッドチーム。本文に旧 `quant-deep-research_SUMMARY` 相当のメタ節を統合済み |
| `deep-research/` | 汎用ディープリサーチ（`quant-deep-research` との使い分けは各 `SKILL.md` 参照） |
| `ff5-macro-analysis/` | `scripts/01`〜`04`（データ取得・グレンジャー・RF・Excel）、`references/data_sources.md` |
| `nonlinear-granger-analysis/` | `scripts/01`〜`05`（サブサンプル・TAR・カーネル・ローリング・レポート） |
| `japan-stock-analyzer/` | `SKILL.md`（ブラウザ/IR バンク前提の手順） |
| `corporate-finance-analysis/` | ルート `SKILL.md` + `skills/` 配下に比率・比較・信用・CF 予測など（各サブスキルに `GUIDE.md`・`scripts/*.py`） |
| `securities-analysis/` | ルート `SKILL.md` + `skills/dcf-valuation`・`dividend-analysis`・`portfolio-analytics`（スクリプト・参照・サンプル付き） |
| `weekly-report/` | `SKILL.md`、`SKILL_for_Claude.md`、`template_weekly_report.docx`、`update_wordcount.py` |
| `qa-pptx-generator/` | `SKILL.md`、`scripts/generate_qa_pptx.py`、`templates/template.pptx` |

## Claude Code での使い方

各サブフォルダを、プロジェクトの `.claude/skills/` 以下にコピーする（ネストした `corporate-finance-analysis/` や `securities-analysis/` は **フォルダごと**コピーする）。

```text
.claude/skills/
├── ff5-macro-analysis/
│   ├── SKILL.md
│   ├── scripts/
│   └── references/
├── corporate-finance-analysis/
│   ├── SKILL.md
│   └── skills/
│       └── ...
└── ...
```

## Claude.ai（ZIP アップロード）

ZIP の直下に **スキル名フォルダ**があり、その中に `SKILL.md` がある形式にする。`corporate-finance-analysis` のように `skills/` が深くなる場合も、**トップのフォルダ単位**で ZIP する。

## 作業ディレクトリでのパス

- **FF5 / 非線形**: プロジェクトルートに `data/`・`results/`・`outputs/` を用意し、`SKILL.md` のコマンド例どおりに実行する。
- **週次報告**: `template_weekly_report.docx` とスキル内スクリプトのパスを `SKILL.md` に従って指定する。

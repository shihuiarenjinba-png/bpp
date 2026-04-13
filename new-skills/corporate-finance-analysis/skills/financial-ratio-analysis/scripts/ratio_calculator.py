#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
財務比率分析スクリプト
企業の財務諸表データから包括的な財務比率を計算し、Excelレポートで出力します。
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as XLImage
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import fontmanager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 日本語フォント設定
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FinancialRatioAnalyzer:
    """財務比率分析のメインクラス"""

    # 業界ベンチマーク値（参考値）
    BENCHMARKS = {
        '収益性': {
            'ROE': {'A': (0.15, 1.0), 'B': (0.10, 0.15), 'C': (0.05, 0.10), 'D': (0.0, 0.05), 'E': (-1.0, 0.0)},
            'ROA': {'A': (0.08, 1.0), 'B': (0.05, 0.08), 'C': (0.02, 0.05), 'D': (0.0, 0.02), 'E': (-1.0, 0.0)},
            '売上高営業利益率': {'A': (0.15, 1.0), 'B': (0.10, 0.15), 'C': (0.05, 0.10), 'D': (0.0, 0.05), 'E': (-1.0, 0.0)},
            '売上高純利益率': {'A': (0.10, 1.0), 'B': (0.05, 0.10), 'C': (0.02, 0.05), 'D': (0.0, 0.02), 'E': (-1.0, 0.0)},
            'EBITDA_margin': {'A': (0.20, 1.0), 'B': (0.15, 0.20), 'C': (0.10, 0.15), 'D': (0.05, 0.10), 'E': (-1.0, 0.05)},
        },
        '安全性': {
            '流動比率': {'A': (1.5, 10.0), 'B': (1.2, 1.5), 'C': (1.0, 1.2), 'D': (0.8, 1.0), 'E': (0.0, 0.8)},
            '当座比率': {'A': (1.0, 10.0), 'B': (0.8, 1.0), 'C': (0.6, 0.8), 'D': (0.4, 0.6), 'E': (0.0, 0.4)},
            '自己資本比率': {'A': (0.50, 1.0), 'B': (0.40, 0.50), 'C': (0.30, 0.40), 'D': (0.20, 0.30), 'E': (0.0, 0.20)},
            'DE_ratio': {'A': (0.0, 1.0), 'B': (1.0, 1.5), 'C': (1.5, 2.0), 'D': (2.0, 3.0), 'E': (3.0, 10.0)},
            'interest_coverage': {'A': (5.0, 100.0), 'B': (3.0, 5.0), 'C': (1.5, 3.0), 'D': (1.0, 1.5), 'E': (0.0, 1.0)},
        },
        '効率性': {
            '総資産回転率': {'A': (1.5, 10.0), 'B': (1.0, 1.5), 'C': (0.7, 1.0), 'D': (0.4, 0.7), 'E': (0.0, 0.4)},
            '棚卸資産回転率': {'A': (8.0, 100.0), 'B': (5.0, 8.0), 'C': (3.0, 5.0), 'D': (1.5, 3.0), 'E': (0.0, 1.5)},
            '売上債権回転日数': {'A': (0.0, 30.0), 'B': (30.0, 45.0), 'C': (45.0, 60.0), 'D': (60.0, 90.0), 'E': (90.0, 365.0)},
            'CCC': {'A': (-100.0, 0.0), 'B': (0.0, 30.0), 'C': (30.0, 60.0), 'D': (60.0, 90.0), 'E': (90.0, 365.0)},
        },
        '市場評価': {
            'PER': {'A': (10.0, 15.0), 'B': (15.0, 20.0), 'C': (20.0, 25.0), 'D': (25.0, 35.0), 'E': (35.0, 100.0)},
            'PBR': {'A': (0.5, 1.5), 'B': (1.5, 2.0), 'C': (2.0, 2.5), 'D': (2.5, 3.5), 'E': (3.5, 10.0)},
            'EV_EBITDA': {'A': (5.0, 12.0), 'B': (12.0, 15.0), 'C': (15.0, 18.0), 'D': (18.0, 25.0), 'E': (25.0, 100.0)},
            '配当利回り': {'A': (0.02, 0.08), 'B': (0.008, 0.02), 'C': (0.003, 0.008), 'D': (0.0, 0.003), 'E': (-0.05, 0.0)},
            'PSR': {'A': (0.5, 2.0), 'B': (2.0, 3.0), 'C': (3.0, 4.0), 'D': (4.0, 6.0), 'E': (6.0, 20.0)},
        }
    }

    def __init__(self, input_path: str, output_path: str = 'financial_analysis_report.xlsx'):
        """
        初期化

        Args:
            input_path: 入力ファイル（CSV/XLSX）のパス
            output_path: 出力ファイル（Excel）のパス
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, pd.DataFrame] = {}
        self.charts = {}

    def load_data(self) -> bool:
        """入力ファイルを読み込む"""
        try:
            if self.input_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(self.input_path)
            elif self.input_path.suffix.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(self.input_path)
            else:
                logger.error(f"非対応のファイル形式: {self.input_path.suffix}")
                return False

            logger.info(f"データ読み込み成功: {len(self.data)} 行")
            return True
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
            return False

    def validate_data(self) -> bool:
        """入力データの妥当性を検証"""
        required_columns = [
            '売上高', '営業利益', '純利益', '総資産', '流動資産',
            '流動負債', '在庫', '売上債権', '買入債務', '自己資本', '負債'
        ]

        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            logger.error(f"必須カラムが見つかりません: {missing}")
            return False

        # 数値型に変換
        for col in required_columns:
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            except Exception as e:
                logger.error(f"カラム '{col}' の数値変換に失敗: {e}")
                return False

        logger.info("データ検証成功")
        return True

    def calculate_profitability_ratios(self) -> pd.DataFrame:
        """収益性指標を計算"""
        df = pd.DataFrame()

        # 基本情報
        df['企業名'] = self.data.get('企業名', ['企業' + str(i) for i in range(len(self.data))])

        # 各指標の計算
        df['ROE'] = self.data['純利益'] / self.data['自己資本']
        df['ROA'] = self.data['純利益'] / self.data['総資産']
        df['売上高営業利益率'] = self.data['営業利益'] / self.data['売上高']
        df['売上高純利益率'] = self.data['純利益'] / self.data['売上高']

        # EPS（利益剰余金や株式数が必要だが、簡略版）
        if '株式数' in self.data.columns:
            df['EPS'] = self.data['純利益'] / self.data['株式数']
        else:
            df['EPS'] = np.nan

        # EBITDA margin
        if 'EBITDA' in self.data.columns:
            df['EBITDA_margin'] = self.data['EBITDA'] / self.data['売上高']
        else:
            df['EBITDA_margin'] = np.nan

        # 無限値とNaNの処理
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def calculate_safety_ratios(self) -> pd.DataFrame:
        """安全性指標を計算"""
        df = pd.DataFrame()

        df['企業名'] = self.data.get('企業名', ['企業' + str(i) for i in range(len(self.data))])

        # 流動比率
        df['流動比率'] = self.data['流動資産'] / self.data['流動負債']

        # 当座比率
        df['当座比率'] = (self.data['流動資産'] - self.data['在庫']) / self.data['流動負債']

        # 自己資本比率
        df['自己資本比率'] = self.data['自己資本'] / self.data['総資産']

        # D/Eレシオ
        df['DE_ratio'] = self.data['負債'] / self.data['自己資本']

        # インタレストカバレッジレシオ（営業利益 / 支払利息）
        # 簡略版：支払利息 = 負債 * 平均金利（仮定：3%）
        interest_expense = self.data['負債'] * 0.03
        df['interest_coverage'] = self.data['営業利益'] / (interest_expense + 1e-9)

        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def calculate_efficiency_ratios(self) -> pd.DataFrame:
        """効率性指標を計算"""
        df = pd.DataFrame()

        df['企業名'] = self.data.get('企業名', ['企業' + str(i) for i in range(len(self.data))])

        # 総資産回転率
        df['総資産回転率'] = self.data['売上高'] / self.data['総資産']

        # 棚卸資産回転率
        df['棚卸資産回転率'] = self.data['売上高'] / (self.data['在庫'] + 1e-9)

        # 売上債権回転日数
        df['売上債権回転日数'] = (self.data['売上債権'] / self.data['売上高']) * 365

        # 買入債務回転日数（簡略版：売上原価の概算）
        cost_of_goods_sold = self.data['売上高'] * 0.6  # 仮定
        df['買入債務回転日数'] = (self.data['買入債務'] / cost_of_goods_sold) * 365

        # キャッシュコンバージョンサイクル (CCC)
        df['CCC'] = df['売上債権回転日数'] + df['買入債務回転日数'] - \
                    ((self.data['在庫'] / (self.data['売上高'] * 0.6)) * 365)

        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def calculate_market_ratios(self) -> pd.DataFrame:
        """市場評価指標を計算"""
        df = pd.DataFrame()

        df['企業名'] = self.data.get('企業名', ['企業' + str(i) for i in range(len(self.data))])

        # PER
        if '株価' in self.data.columns and '株式数' in self.data.columns:
            market_cap = self.data['株価'] * self.data['株式数']
            df['PER'] = market_cap / (self.data['純利益'] + 1e-9)
        else:
            df['PER'] = np.nan

        # PBR
        if '株価' in self.data.columns and '株式数' in self.data.columns:
            market_cap = self.data['株価'] * self.data['株式数']
            df['PBR'] = market_cap / (self.data['自己資本'] + 1e-9)
        else:
            df['PBR'] = np.nan

        # EV/EBITDA
        if 'EBITDA' in self.data.columns and '株価' in self.data.columns and '株式数' in self.data.columns:
            market_cap = self.data['株価'] * self.data['株式数']
            enterprise_value = market_cap + self.data['負債']
            df['EV_EBITDA'] = enterprise_value / (self.data['EBITDA'] + 1e-9)
        else:
            df['EV_EBITDA'] = np.nan

        # 配当利回り
        if '配当金' in self.data.columns and '株価' in self.data.columns:
            df['配当利回り'] = self.data['配当金'] / (self.data['株価'] + 1e-9)
        else:
            df['配当利回り'] = np.nan

        # PSR
        if '株価' in self.data.columns and '株式数' in self.data.columns:
            market_cap = self.data['株価'] * self.data['株式数']
            df['PSR'] = market_cap / (self.data['売上高'] + 1e-9)
        else:
            df['PSR'] = np.nan

        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def calculate_dupont_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """デュポン分解を計算（3段階と5段階）"""
        # 3段階分解
        dupont_3 = pd.DataFrame()
        dupont_3['企業名'] = self.data.get('企業名', ['企業' + str(i) for i in range(len(self.data))])

        # ROE = 純利益率 × 総資産回転率 × 財務レバレッジ
        dupont_3['純利益率'] = self.data['純利益'] / self.data['売上高']
        dupont_3['総資産回転率'] = self.data['売上高'] / self.data['総資産']
        dupont_3['財務レバレッジ'] = self.data['総資産'] / self.data['自己資本']
        dupont_3['ROE (計算値)'] = dupont_3['純利益率'] * dupont_3['総資産回転率'] * dupont_3['財務レバレッジ']

        # 5段階分解
        dupont_5 = pd.DataFrame()
        dupont_5['企業名'] = self.data.get('企業名', ['企業' + str(i) for i in range(len(self.data))])

        # ROE = 営業利益率 × 売上高営業利益率 × 総資産回転率 × (1 - 税率) × 財務レバレッジ
        # 簡略化: 税率 = 30%
        dupont_5['営業利益率'] = self.data['営業利益'] / self.data['売上高']
        dupont_5['純利益率'] = self.data['純利益'] / self.data['営業利益']
        dupont_5['総資産回転率'] = self.data['売上高'] / self.data['総資産']
        dupont_5['財務レバレッジ'] = self.data['総資産'] / self.data['自己資本']
        dupont_5['ROE (計算値)'] = dupont_5['営業利益率'] * dupont_5['純利益率'] * \
                                   dupont_5['総資産回転率'] * dupont_5['財務レバレッジ']

        return dupont_3.replace([np.inf, -np.inf], np.nan), \
               dupont_5.replace([np.inf, -np.inf], np.nan)

    def grade_ratio(self, value: float, category: str, ratio_name: str) -> str:
        """比率に基づいて評級を返す"""
        if pd.isna(value):
            return '-'

        benchmarks = self.BENCHMARKS.get(category, {}).get(ratio_name, {})

        if not benchmarks:
            return '-'

        for grade in ['A', 'B', 'C', 'D', 'E']:
            if grade in benchmarks:
                min_val, max_val = benchmarks[grade]
                if min_val <= value <= max_val:
                    return grade

        return 'E'

    def analyze(self) -> bool:
        """全体的な分析を実行"""
        if not self.load_data():
            return False

        if not self.validate_data():
            return False

        logger.info("各種比率を計算中...")

        # 各比率を計算
        self.results['収益性指標'] = self.calculate_profitability_ratios()
        self.results['安全性指標'] = self.calculate_safety_ratios()
        self.results['効率性指標'] = self.calculate_efficiency_ratios()
        self.results['市場評価指標'] = self.calculate_market_ratios()
        self.results['デュポン_3段階'], self.results['デュポン_5段階'] = self.calculate_dupont_analysis()

        logger.info("分析完了")
        return True

    def create_radar_chart(self) -> Optional[str]:
        """レーダーチャートを作成"""
        try:
            import io

            # 正規化された値を使用
            prof_df = self.results['収益性指標'].iloc[0]
            safe_df = self.results['安全性指標'].iloc[0]
            eff_df = self.results['効率性指標'].iloc[0]

            # 各カテゴリの平均スコアを計算（0-100のスケール）
            categories = ['収益性', '安全性', '効率性']

            # 簡略化：各指標のグレードをスコアに変換
            def grade_to_score(grade: str) -> float:
                grade_scores = {'A': 100, 'B': 75, 'C': 50, 'D': 25, 'E': 0, '-': 0}
                return grade_scores.get(grade, 0)

            scores = []
            for cat in categories:
                cat_df = self.results[cat + '指標'].iloc[0]
                if cat == '収益性':
                    grade = self.grade_ratio(cat_df['ROE'], '収益性', 'ROE')
                elif cat == '安全性':
                    grade = self.grade_ratio(cat_df['自己資本比率'], '安全性', '自己資本比率')
                else:
                    grade = self.grade_ratio(cat_df['総資産回転率'], '効率性', '総資産回転率')
                scores.append(grade_to_score(grade))

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            scores_plot = scores + [scores[0]]
            angles_plot = angles + [angles[0]]

            ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, label='スコア', color='#1f77b4')
            ax.fill(angles_plot, scores_plot, alpha=0.25, color='#1f77b4')
            ax.set_xticks(angles)
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.grid(True)
            ax.set_title('財務比率総合評価', fontsize=12, fontweight='bold', pad=20)

            # 画像をバイナリに保存
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()

            # 一時ファイルに保存
            chart_path = self.output_path.parent / 'radar_chart.png'
            with open(chart_path, 'wb') as f:
                f.write(buf.getvalue())

            return str(chart_path)
        except Exception as e:
            logger.warning(f"レーダーチャート作成エラー: {e}")
            return None

    def export_to_excel(self) -> bool:
        """結果をExcelに出力"""
        try:
            logger.info(f"Excelファイルを作成中: {self.output_path}")

            with pd.ExcelWriter(self.output_path, engine='openpyxl') as writer:
                # サマリーシート
                summary_df = pd.DataFrame({
                    '指標カテゴリ': ['収益性 (ROE)', '安全性 (自己資本比率)', '効率性 (総資産回転率)'],
                    '値': [
                        self.results['収益性指標'].iloc[0]['ROE'],
                        self.results['安全性指標'].iloc[0]['自己資本比率'],
                        self.results['効率性指標'].iloc[0]['総資産回転率']
                    ],
                    '評級': [
                        self.grade_ratio(self.results['収益性指標'].iloc[0]['ROE'], '収益性', 'ROE'),
                        self.grade_ratio(self.results['安全性指標'].iloc[0]['自己資本比率'], '安全性', '自己資本比率'),
                        self.grade_ratio(self.results['効率性指標'].iloc[0]['総資産回転率'], '効率性', '総資産回転率')
                    ]
                })
                summary_df.to_excel(writer, sheet_name='サマリー', index=False)

                # 各比率をシートに出力
                self.results['収益性指標'].to_excel(writer, sheet_name='収益性指標', index=False)
                self.results['安全性指標'].to_excel(writer, sheet_name='安全性指標', index=False)
                self.results['効率性指標'].to_excel(writer, sheet_name='効率性指標', index=False)
                self.results['市場評価指標'].to_excel(writer, sheet_name='市場評価指標', index=False)

                # デュポン分解
                self.results['デュポン_3段階'].to_excel(writer, sheet_name='デュポン分解', index=False, startrow=0)
                self.results['デュポン_5段階'].to_excel(writer, sheet_name='デュポン分解', index=False,
                                                      startrow=len(self.results['デュポン_3段階']) + 3)

            logger.info(f"ファイル出力成功: {self.output_path}")
            return True
        except Exception as e:
            logger.error(f"Excel出力エラー: {e}")
            return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='財務比率分析スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='入力ファイル（CSV/XLSX）のパス'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='financial_analysis_report.xlsx',
        help='出力ファイル（Excel）のパス（デフォルト: financial_analysis_report.xlsx）'
    )

    args = parser.parse_args()

    analyzer = FinancialRatioAnalyzer(args.input, args.output)

    if not analyzer.analyze():
        logger.error("分析に失敗しました")
        sys.exit(1)

    if not analyzer.export_to_excel():
        logger.error("Excelへの出力に失敗しました")
        sys.exit(1)

    logger.info("処理完了")
    print(f"\n成功: {analyzer.output_path} にファイルが出力されました")


if __name__ == '__main__':
    main()

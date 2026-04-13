#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ポートフォリオ分析スクリプト

投資ポートフォリオのリスク・リターン分析を実施し、
詳細なExcelレポートと可視化を生成します。

使用方法:
    python portfolio_analyzer.py --input input.csv --output output.xlsx [--rf-rate 0.02] [--periods 252]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # バックエンド設定（ファイル出力用）
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as XLImage
from openpyxl.worksheet.table import Table, TableStyleInfo


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioAnalyzer:
    """ポートフォリオ分析を実施するメインクラス"""

    def __init__(self, risk_free_rate: float = 0.02, periods: int = 252):
        """
        初期化

        Args:
            risk_free_rate: リスクフリーレート（年率、デフォルト0.02=2%）
            periods: 年間営業日数（デフォルト252日）
        """
        self.risk_free_rate = risk_free_rate
        self.periods = periods
        self.df = None
        self.holdings = []
        self.metrics = {}
        self.charts = {}

    def load_data(self, input_path: str) -> None:
        """
        CSVファイルをロード

        Args:
            input_path: 入力CSVファイルパス

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: 必要な列が見つからない場合
        """
        logger.info(f"CSVファイルをロード: {input_path}")

        if not Path(input_path).exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {input_path}")

        try:
            self.df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(input_path, encoding='shift_jis')

        # 必要な列の確認
        required_columns = ['銘柄', '取得価格', '数量', '現在価格', 'セクター']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"不正な列が見つかりました。以下の列がありません: {missing_columns}")

        logger.info(f"データロード完了: {len(self.df)}件の銘柄")

    def validate_data(self) -> None:
        """データの妥当性をチェック"""
        logger.info("データの妥当性をチェック中...")

        # 数値列の型確認・変換
        numeric_cols = ['取得価格', '数量', '現在価格']
        for col in numeric_cols:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except ValueError as e:
                raise ValueError(f"列 '{col}' に数値以外のデータが含まれています: {e}")

        # 負数チェック
        if (self.df['取得価格'] <= 0).any() or (self.df['現在価格'] <= 0).any():
            raise ValueError("価格は正の数である必要があります")
        if (self.df['数量'] <= 0).any():
            raise ValueError("数量は正の数である必要があります")

        logger.info("データ検証完了")

    def calculate_holding_metrics(self) -> None:
        """各銘柄のメトリクスを計算"""
        logger.info("銘柄別メトリクスを計算中...")

        self.df['投資金額'] = self.df['取得価格'] * self.df['数量']
        self.df['現在評価額'] = self.df['現在価格'] * self.df['数量']
        self.df['取得時からの利益'] = self.df['現在評価額'] - self.df['投資金額']
        self.df['リターン率'] = (self.df['現在価格'] - self.df['取得価格']) / self.df['取得価格']

        # ポートフォリオ全体の指標
        total_invested = self.df['投資金額'].sum()
        total_value = self.df['現在評価額'].sum()

        self.df['ウェイト'] = self.df['投資金額'] / total_invested
        self.df['加重リターン'] = self.df['ウェイト'] * self.df['リターン率']

        self.metrics['総投資金額'] = total_invested
        self.metrics['現在の総評価額'] = total_value
        self.metrics['全体利益'] = total_value - total_invested
        self.metrics['全体リターン率'] = (total_value - total_invested) / total_invested
        self.metrics['ポートフォリオリターン'] = self.df['加重リターン'].sum()

        logger.info("銘柄別メトリクス計算完了")

    def calculate_risk_metrics(self) -> None:
        """リスク指標を計算"""
        logger.info("リスク指標を計算中...")

        # リターン率の標準偏差（各銘柄）
        returns = self.df['リターン率'].values

        # ポートフォリオ全体の統計
        portfolio_return = self.metrics['ポートフォリオリターン']
        weighted_returns = self.df['加重リターン'].values

        # 標準偏差（リスク）
        self.metrics['ポートフォリオ標準偏差'] = np.std(weighted_returns)
        self.metrics['ポートフォリオ分散'] = np.var(weighted_returns)

        # シャープレシオ = (ポートフォリオリターン - リスクフリーレート) / 標準偏差
        excess_return = portfolio_return - self.risk_free_rate
        if self.metrics['ポートフォリオ標準偏差'] > 0:
            self.metrics['シャープレシオ'] = excess_return / self.metrics['ポートフォリオ標準偏差']
        else:
            self.metrics['シャープレシオ'] = 0

        # ソルティーノレシオ（下方偏差を考慮）
        negative_returns = weighted_returns[weighted_returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
            if downside_deviation > 0:
                self.metrics['ソルティーノレシオ'] = excess_return / downside_deviation
            else:
                self.metrics['ソルティーノレシオ'] = 0
        else:
            self.metrics['ソルティーノレシオ'] = float('inf')

        # 最大ドローダウン（シミュレーション）
        self.metrics['最大ドローダウン'] = self._calculate_max_drawdown()

        logger.info("リスク指標計算完了")

    def _calculate_max_drawdown(self) -> float:
        """
        最大ドローダウンを計算

        Returns:
            最大ドローダウン（負の値で返される）
        """
        # 各銘柄のリターンから累積リターンをシミュレート
        returns = self.df['リターン率'].values
        weights = self.df['ウェイト'].values

        # 加重リターンの累積
        cumulative_return = np.cumprod(1 + (returns * weights))
        running_max = np.maximum.accumulate(cumulative_return)
        drawdown = (cumulative_return - running_max) / running_max
        max_drawdown = np.min(drawdown)

        return max_drawdown

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        相関係数行列を計算

        Returns:
            相関係数行列 DataFrame
        """
        logger.info("相関係数行列を計算中...")

        returns_matrix = self.df[['銘柄', 'リターン率']].set_index('銘柄').T

        # 銘柄数が少ない場合の処理
        if len(self.df) < 2:
            logger.warning("銘柄数が2未満のため、相関係数は計算できません")
            return pd.DataFrame()

        correlation_matrix = returns_matrix.corr()

        logger.info("相関係数行列計算完了")
        return correlation_matrix

    def calculate_sector_metrics(self) -> None:
        """セクター別メトリクスを計算"""
        logger.info("セクター別メトリクスを計算中...")

        sector_data = self.df.groupby('セクター').agg({
            '投資金額': 'sum',
            '現在評価額': 'sum',
            '取得時からの利益': 'sum'
        }).reset_index()

        sector_data['セクター構成比'] = sector_data['投資金額'] / sector_data['投資金額'].sum()
        sector_data['セクターリターン'] = (
            (sector_data['現在評価額'] - sector_data['投資金額']) /
            sector_data['投資金額']
        )

        # ハーフィンダール指数（集中リスク指標）
        herfindahl_index = (sector_data['セクター構成比'] ** 2).sum()
        self.metrics['ハーフィンダール指数'] = herfindahl_index
        self.metrics['セクター集中度'] = '高' if herfindahl_index > 0.25 else (
            '中' if herfindahl_index > 0.15 else '低'
        )

        self.metrics['セクター別データ'] = sector_data
        logger.info("セクター別メトリクス計算完了")

    def generate_charts(self, output_dir: str = '.') -> None:
        """
        分析チャートを生成

        Args:
            output_dir: 出力ディレクトリ
        """
        logger.info("チャートを生成中...")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # 1. ポートフォリオ配分比率（パイチャート）
        self._create_allocation_pie_chart(output_dir)

        # 2. リターン棒グラフ
        self._create_returns_bar_chart(output_dir)

        # 3. 相関係数ヒートマップ
        if len(self.df) > 1:
            self._create_correlation_heatmap(output_dir)

        logger.info("チャート生成完了")

    def _create_allocation_pie_chart(self, output_dir: str) -> None:
        """配分比率パイチャートを生成"""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.df)))
        wedges, texts, autotexts = ax.pie(
            self.df['投資金額'],
            labels=self.df['銘柄'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 10, 'family': 'DejaVu Sans'}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('ポートフォリオ配分比率', fontsize=14, fontweight='bold',
                    fontproperties='SimHei')

        plt.tight_layout()
        chart_path = Path(output_dir) / 'allocation_pie.png'
        plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
        plt.close()

        self.charts['allocation_pie'] = str(chart_path)
        logger.debug(f"パイチャート保存: {chart_path}")

    def _create_returns_bar_chart(self, output_dir: str) -> None:
        """リターン棒グラフを生成"""
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['green' if r > 0 else 'red' for r in self.df['リターン率']]
        ax.bar(self.df['銘柄'], self.df['リターン率'] * 100, color=colors, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('リターン率 (%)', fontsize=12, fontproperties='SimHei')
        ax.set_xlabel('銘柄', fontsize=12, fontproperties='SimHei')
        ax.set_title('銘柄別リターン率', fontsize=14, fontweight='bold',
                    fontproperties='SimHei')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        chart_path = Path(output_dir) / 'returns_bar.png'
        plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
        plt.close()

        self.charts['returns_bar'] = str(chart_path)
        logger.debug(f"棒グラフ保存: {chart_path}")

    def _create_correlation_heatmap(self, output_dir: str) -> None:
        """相関係数ヒートマップを生成"""
        correlation_matrix = self.calculate_correlation_matrix()

        if correlation_matrix.empty:
            logger.warning("相関係数行列が空のため、ヒートマップを生成できません")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            cbar_kws={'label': '相関係数'}
        )

        ax.set_title('銘柄間相関係数ヒートマップ', fontsize=14, fontweight='bold',
                    fontproperties='SimHei')

        plt.tight_layout()
        chart_path = Path(output_dir) / 'correlation_heatmap.png'
        plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
        plt.close()

        self.charts['correlation_heatmap'] = str(chart_path)
        logger.debug(f"ヒートマップ保存: {chart_path}")

    def export_to_excel(self, output_path: str) -> None:
        """
        分析結果をExcelにエクスポート

        Args:
            output_path: 出力Excelファイルパス
        """
        logger.info(f"Excelファイルをエクスポート: {output_path}")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # シート1: サマリー
            self._write_summary_sheet(writer)

            # シート2: 個別銘柄分析
            self._write_holdings_sheet(writer)

            # シート3: リスク指標
            self._write_risk_metrics_sheet(writer)

            # シート4: 相関行列
            if len(self.df) > 1:
                self._write_correlation_sheet(writer)

            # シート5: セクター配分
            self._write_sector_sheet(writer)

        logger.info("Excelファイル出力完了")

    def _write_summary_sheet(self, writer) -> None:
        """サマリーシートを作成"""
        summary_data = {
            '項目': [
                '総投資金額',
                '現在の総評価額',
                '全体利益',
                '全体リターン率',
                'ポートフォリオリターン',
                'ポートフォリオ標準偏差',
                'ポートフォリオ分散',
                'シャープレシオ',
                'ソルティーノレシオ',
                '最大ドローダウン',
                'ハーフィンダール指数',
                'セクター集中度'
            ],
            '値': [
                f"¥{self.metrics['総投資金額']:,.0f}",
                f"¥{self.metrics['現在の総評価額']:,.0f}",
                f"¥{self.metrics['全体利益']:,.0f}",
                f"{self.metrics['全体リターン率']:.2%}",
                f"{self.metrics['ポートフォリオリターン']:.2%}",
                f"{self.metrics['ポートフォリオ標準偏差']:.4f}",
                f"{self.metrics['ポートフォリオ分散']:.4f}",
                f"{self.metrics['シャープレシオ']:.4f}",
                f"{self.metrics['ソルティーノレシオ']:.4f}",
                f"{self.metrics['最大ドローダウン']:.2%}",
                f"{self.metrics['ハーフィンダール指数']:.4f}",
                self.metrics['セクター集中度']
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='サマリー', index=False)

    def _write_holdings_sheet(self, writer) -> None:
        """個別銘柄分析シートを作成"""
        holdings_df = self.df[[
            '銘柄',
            '取得価格',
            '数量',
            '現在価格',
            '投資金額',
            '現在評価額',
            '取得時からの利益',
            'リターン率',
            'ウェイト',
            'セクター'
        ]].copy()

        # フォーマット
        holdings_df['リターン率'] = holdings_df['リターン率'].apply(lambda x: f'{x:.2%}')
        holdings_df['ウェイト'] = holdings_df['ウェイト'].apply(lambda x: f'{x:.2%}')

        holdings_df.to_excel(writer, sheet_name='個別銘柄分析', index=False)

    def _write_risk_metrics_sheet(self, writer) -> None:
        """リスク指標シートを作成"""
        risk_data = {
            'リスク指標': [
                'シャープレシオ',
                'ソルティーノレシオ',
                '最大ドローダウン',
                'ポートフォリオ標準偏差',
                'ポートフォリオ分散'
            ],
            '値': [
                f"{self.metrics['シャープレシオ']:.4f}",
                f"{self.metrics['ソルティーノレシオ']:.4f}",
                f"{self.metrics['最大ドローダウン']:.2%}",
                f"{self.metrics['ポートフォリオ標準偏差']:.4f}",
                f"{self.metrics['ポートフォリオ分散']:.4f}"
            ],
            '説明': [
                'リスク調整後リターン（高いほど良い）',
                '下方リスク考慮版シャープレシオ',
                '最悪の場合の損失率',
                'ポートフォリオの変動性',
                'ポートフォリオの分散（標準偏差の二乗）'
            ]
        }

        risk_df = pd.DataFrame(risk_data)
        risk_df.to_excel(writer, sheet_name='リスク指標', index=False)

    def _write_correlation_sheet(self, writer) -> None:
        """相関行列シートを作成"""
        correlation_matrix = self.calculate_correlation_matrix()

        if not correlation_matrix.empty:
            correlation_matrix.to_excel(writer, sheet_name='相関行列')

    def _write_sector_sheet(self, writer) -> None:
        """セクター配分シートを作成"""
        sector_df = self.metrics['セクター別データ'].copy()

        sector_df['セクター構成比'] = sector_df['セクター構成比'].apply(lambda x: f'{x:.2%}')
        sector_df['セクターリターン'] = sector_df['セクターリターン'].apply(lambda x: f'{x:.2%}')

        sector_df.to_excel(writer, sheet_name='セクター配分', index=False)

    def run(self, input_path: str, output_path: str) -> None:
        """
        分析全体を実行

        Args:
            input_path: 入力CSVファイルパス
            output_path: 出力Excelファイルパス
        """
        try:
            logger.info("=== ポートフォリオ分析を開始 ===")

            self.load_data(input_path)
            self.validate_data()
            self.calculate_holding_metrics()
            self.calculate_risk_metrics()
            self.calculate_correlation_matrix()
            self.calculate_sector_metrics()

            # チャート生成（出力ファイルと同じディレクトリ）
            output_dir = Path(output_path).parent
            self.generate_charts(str(output_dir))

            self.export_to_excel(output_path)

            logger.info("=== ポートフォリオ分析が正常に完了しました ===")
            logger.info(f"出力ファイル: {output_path}")

        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='ポートフォリオ分析スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
例:
  python portfolio_analyzer.py --input portfolio.csv --output report.xlsx
  python portfolio_analyzer.py --input portfolio.csv --output report.xlsx --rf-rate 0.03 --periods 252
        '''
    )

    parser.add_argument(
        '--input',
        required=True,
        help='入力CSVファイルのパス'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='出力Excelファイルのパス'
    )
    parser.add_argument(
        '--rf-rate',
        type=float,
        default=0.02,
        help='リスクフリーレート（年率、デフォルト0.02=2%）'
    )
    parser.add_argument(
        '--periods',
        type=int,
        default=252,
        help='年間営業日数（デフォルト252日）'
    )

    args = parser.parse_args()

    analyzer = PortfolioAnalyzer(
        risk_free_rate=args.rf_rate,
        periods=args.periods
    )
    analyzer.run(args.input, args.output)


if __name__ == '__main__':
    main()

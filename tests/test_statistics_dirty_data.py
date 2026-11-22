import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


portfolio_mod = load_module(
    "portfolio_analyzer",
    "new-skills/securities-analysis/skills/portfolio-analytics/scripts/portfolio_analyzer.py",
)
linear = load_module(
    "linear_granger",
    "new-skills/ff5-macro-analysis/scripts/02_granger_analysis.py",
)
threshold = load_module(
    "threshold_granger",
    "new-skills/nonlinear-granger-analysis/scripts/02_threshold_granger.py",
)
kernel = load_module(
    "kernel_granger",
    "new-skills/nonlinear-granger-analysis/scripts/03_kernel_granger.py",
)


class PortfolioAnalyticsDirtyTests(unittest.TestCase):
    def write_csv(self, text, encoding="utf-8-sig"):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.close()
        Path(tmp.name).write_text(text, encoding=encoding)
        return tmp.name

    def test_lightly_dirty_portfolio_is_normalized(self):
        path = self.write_csv(
            " 銘柄 , 取得価格 , 数量 , 現在価格 , セクター , メモ\n"
            'AAPL,"1,000",10,"1,100",Tech,ok\n'
            'MSFT,"2,000",5,"1,900",Tech,note\n'
        )
        a = portfolio_mod.PortfolioAnalyzer()
        a.load_data(path)
        a.validate_data()
        self.assertEqual(list(a.df["銘柄"]), ["AAPL", "MSFT"])
        self.assertEqual(float(a.df.loc[0, "取得価格"]), 1000.0)
        self.assertEqual(float(a.df.loc[1, "数量"]), 5.0)

    def test_blank_nonfinite_negative_and_duplicate_values_fail_closed(self):
        cases = [
            (
                "銘柄,取得価格,数量,現在価格,セクター\nAAPL,,10,1100,Tech\n",
                "空欄・非数値・非有限値",
            ),
            (
                "銘柄,取得価格,数量,現在価格,セクター\nAAPL,1000,inf,1100,Tech\n",
                "空欄・非数値・非有限値",
            ),
            (
                "銘柄,取得価格,数量,現在価格,セクター\nAAPL,-1000,10,1100,Tech\n",
                "価格は正",
            ),
            (
                "銘柄,取得価格,数量,現在価格,セクター\nAAPL,1000,10,1100,Tech\nAAPL,1200,5,1300,Tech\n",
                "重複",
            ),
            (
                "銘柄,取得価格,数量,現在価格,セクター\nAAPL,1000,10,1100,\n",
                "セクター",
            ),
        ]
        for text, pattern in cases:
            a = portfolio_mod.PortfolioAnalyzer()
            path = self.write_csv(text)
            a.load_data(path)
            with self.assertRaisesRegex(ValueError, pattern):
                a.validate_data()


class LinearGrangerDirtyTests(unittest.TestCase):
    def test_inf_and_nan_are_removed_before_granger(self):
        n = 90
        idx = pd.date_range("2000-01-01", periods=n, freq="ME")
        x = pd.Series(np.sin(np.arange(n) / 5), index=idx, name="CPI_growth")
        y = pd.Series(0.3 * x.shift(1).fillna(0) + np.cos(np.arange(n) / 9) * 0.05,
                      index=idx, name="Mkt-RF")
        x.iloc[10] = np.inf
        y.iloc[20] = np.nan
        result = linear.run_granger_test(y, x, lag=2)
        self.assertEqual(result["n_obs"], 88)
        self.assertTrue(math.isfinite(result["f_stat"]) or math.isnan(result["f_stat"]))

    def test_invalid_lag_is_rejected(self):
        s = pd.Series(np.arange(30.0), name="x")
        with self.assertRaisesRegex(ValueError, "max_lag"):
            linear.select_lag_bic(s, s, 0)
        with self.assertRaisesRegex(ValueError, "lag"):
            linear.run_granger_test(s, s, 0)

    def test_empty_significance_table_does_not_divide_by_zero(self):
        out = linear.add_significance_flags(pd.DataFrame(), alpha=0.05)
        self.assertEqual(len(out), 0)
        self.assertIn("Bonferroni補正後α", out.columns)
        self.assertIn("総検定数", out.columns)


class ThresholdGrangerDirtyTests(unittest.TestCase):
    def test_all_missing_or_nonfinite_threshold_returns_empty(self):
        data = pd.DataFrame({
            "DEF_SPREAD": [np.nan, np.inf] * 15,
            "Mkt-RF": np.linspace(0, 1, 30),
            "CPI_growth": np.linspace(1, 2, 30),
        })
        out = threshold.run_threshold_granger(data, "DEF_SPREAD", 3)
        self.assertTrue(out.empty)

    def test_finite_filter_avoids_inf_crash(self):
        n = 60
        x = np.sin(np.arange(n) / 4)
        y = np.roll(x, 1) * 0.2 + np.cos(np.arange(n) / 7) * 0.1
        x[5] = np.inf
        f, p = threshold.granger_f_test(y, x, 2)
        self.assertTrue(math.isfinite(f) or math.isnan(f))
        self.assertTrue(math.isfinite(p) or math.isnan(p))

    def test_invalid_max_lag_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_lag"):
            threshold.run_threshold_granger(pd.DataFrame({"DEF_SPREAD": range(30)}), max_lag=0)


class KernelGrangerDirtyTests(unittest.TestCase):
    def test_nan_inf_are_filtered_and_small_permutation_run_finishes(self):
        n = 45
        x = np.sin(np.arange(n) / 5).astype(float)
        y = (0.3 * np.roll(x, 1) + np.cos(np.arange(n) / 8) * 0.05).astype(float)
        x[3] = np.inf
        y[10] = np.nan
        stat, p = kernel.permutation_test(y, x, lag=1, n_perm=9, seed=1)
        self.assertTrue(math.isfinite(stat) or math.isnan(stat))
        self.assertTrue((0 <= p <= 1) or math.isnan(p))

    def test_run_kernel_handles_nonfinite_frame_and_print_format(self):
        n = 45
        data = pd.DataFrame({
            "Mkt-RF": np.sin(np.arange(n) / 4),
            "CPI_growth": np.cos(np.arange(n) / 6),
        })
        data.loc[5, "Mkt-RF"] = np.inf
        data.loc[7, "CPI_growth"] = np.nan
        out = kernel.run_kernel_granger(data, n_perm=5, lag=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.iloc[0]["観測数"]), 43)

    def test_invalid_kernel_parameters_rejected(self):
        with self.assertRaisesRegex(ValueError, "n_perm"):
            kernel.permutation_test(np.arange(30.0), np.arange(30.0), n_perm=0)
        with self.assertRaisesRegex(ValueError, "lag"):
            kernel.run_kernel_granger(pd.DataFrame(), n_perm=3, lag=0)


if __name__ == "__main__":
    unittest.main()

"""Regression and data-availability checks. Run: python -m unittest discover -s tests -v."""
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cash_model as model


class UpdatedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cls.data, cls.rates = model.read_sources(
                ROOT / 'data/cash_sales.csv', ROOT / 'data/macro_scenario.xlsx',
                ROOT / 'data/price_index.xlsx',
                price_assumptions=ROOT / 'data/price_assumptions.csv')
        cls.fitted = model.fit_model(cls.data.real, cls.data.ks_avg3)

    def test_cleaned_source_and_sample(self):
        self.assertEqual(len(self.data), 45)
        self.assertEqual(int(self.fitted['valid'].sum()), 44)
        self.assertEqual(self.data.index[-1], pd.Timestamp('2026-09-01'))
        self.assertAlmostEqual(self.data.loc['2025-02-01', 'nominal'], 95.10640885120992)
        self.assertAlmostEqual(self.data.loc['2025-04-01', 'nominal'], 80.72142426296058)
        self.assertTrue(np.isnan(self.data.loc['2023-02-01', 'real']))

    def test_updated_coefficients(self):
        np.testing.assert_allclose(self.fitted['coef'], [-4.163516894009227, 2.830075324593157], atol=1e-8)

    def test_annual_forecast_and_units(self):
        monthly, annual = model.forecasts(self.data, self.rates, self.fitted, 51, 282700)
        self.assertEqual(monthly.index[-1], pd.Timestamp('2030-12-01'))
        np.testing.assert_allclose(annual.real_bln_rub,
                                  [382.4475499077576, 387.67014680044826, 319.79238513989355, 289.64683622306814, 289.64683622306814])
        np.testing.assert_allclose(annual.area_mln_m2, annual.real_bln_rub / 282.7)
        self.assertEqual(annual.actual_months.iloc[0], 9)
        self.assertEqual(annual.forecast_months.iloc[0], 3)
        self.assertFalse(annual.provisional.any())

    def test_actual_september_index_and_no_assumptions(self):
        self.assertFalse(self.data.price_index_assumed.any())
        self.assertEqual(self.data.attrs['price_assumptions_applied'], [])
        self.assertAlmostEqual(self.data.loc['2026-09-01', 'price_factor'], 241.72054 / 140.23915)
        strict, _ = model.read_sources(ROOT / 'data/cash_sales.csv',
            ROOT / 'data/macro_scenario.xlsx', ROOT / 'data/price_index.xlsx')
        np.testing.assert_allclose(strict.real, self.data.real, equal_nan=True)

    def test_actual_price_replaces_assumption(self):
        dates = pd.to_datetime(['2026-08-01', '2026-09-01'])
        prices = pd.Series([236.78928, 240.0], index=dates)
        aligned, flags, applied = model.apply_price_assumptions(
            prices, dates, ROOT / 'data/price_assumptions.csv')
        self.assertEqual(aligned.iloc[-1], 240.0)
        self.assertFalse(flags.any())
        self.assertEqual(applied, [])

    def test_assumptions_do_not_fill_other_missing_months(self):
        dates = pd.date_range('2026-08-01', '2026-10-01', freq='MS')
        prices = pd.Series([236.78928], index=dates[:1])
        aligned, flags, _ = model.apply_price_assumptions(
            prices, dates, ROOT / 'data/price_assumptions.csv')
        self.assertTrue(np.isnan(aligned.loc['2026-10-01']))
        self.assertFalse(flags.loc['2026-10-01'])

    def test_strict_mode_rejects_missing_september_price(self):
        with self.assertRaisesRegex(ValueError, 'Missing prices'):
            model.read_sources(ROOT / 'data/cash_sales.csv', ROOT / 'data/macro_scenario.xlsx',
                               ROOT / 'archive/price_index_2026-08.xlsx')

    def test_as_of_august_does_not_use_september(self):
        data, _ = model.read_sources(
            ROOT / 'data/cash_sales.csv', ROOT / 'data/macro_scenario.xlsx',
            ROOT / 'data/price_index.xlsx', as_of='2026-08-01',
            price_assumptions=ROOT / 'data/price_assumptions.csv')
        self.assertFalse(data.price_index_assumed.any())
        self.assertEqual(data.attrs['price_assumptions_applied'], [])
        np.testing.assert_allclose(model.fit_model(data.real, data.ks_avg3)['coef'],
                                   [-3.0330960871104438, 2.7957544823968843], atol=1e-8)

    def test_cash_csv_rejects_duplicate_months(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cash.csv'
            path.write_text('date,nominal_bln_rub\n2026-09-01,30\n2026-09-01,31\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                model.read_cash_series(path)

    def test_legacy_cash_format_remains_readable(self):
        data = model.read_cash_series(ROOT / 'archive/cash_sales_2026-08.xlsx')
        self.assertEqual(data.index[-1], pd.Timestamp('2026-08-01'))
        self.assertAlmostEqual(data.loc['2025-04-01'], 156.74614052413057)


if __name__ == '__main__':
    unittest.main()

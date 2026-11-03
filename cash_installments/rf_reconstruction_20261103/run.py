"""Rebuild the saved snapshot and verify accounting identities and reference outputs."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
names = ['current_monthly.csv', 'current_quarterly.csv',
         'before_logged_corrections_monthly.csv', 'before_logged_corrections_quarterly.csv',
         'corrections_impact.csv', 'keyrate_quarterly.csv',
         'model_sensitivity_OLS.csv', 'model_sensitivity_ARIMA.csv']
reference = {name: pd.read_csv(root / name) for name in names}
for script in ['rebuild.py', 'fit.py']:
    subprocess.run([sys.executable, str(root / script)], check=True)
for name, expected in reference.items():
    pd.testing.assert_frame_equal(pd.read_csv(root / name), expected,
                                  check_exact=False, rtol=1e-6, atol=1e-6)
d = pd.read_csv(root / 'current_monthly.csv')
q = pd.read_csv(root / 'current_quarterly.csv')
assert len(d) == 69 and len(q) == 23
assert not d[['ddu_mln', 'pv_pct', 'cash_th_m2']].isna().any().any()
np.testing.assert_allclose(d.sales_mln, d.mortgage_mln + d.cash_mln)
np.testing.assert_allclose(d.sales_th_m2, d.cash_th_m2 + d.mortgage_mln / d.price_th_rub_m2)
np.testing.assert_allclose(q.MA5, q.cash_th_m2.rolling(5, min_periods=1).mean())
print('Verified: saved results, 69 months, 23 quarters, money/area identities and MA5.')


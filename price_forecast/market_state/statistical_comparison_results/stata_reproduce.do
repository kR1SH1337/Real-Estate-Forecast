version 13.0
clear all
set more off
* Run from the repository root. Not executed in Stata in this environment.
import delimited using "price_forecast/market_state/statistical_comparison_results/model_data.csv", varnames(1) clear
generate tm = monthly(substr(date,1,7), "YM")
format tm %tm
tsset tm

* growth
nl (growth_target = {gc=0.147506430574} + {gar=0.185840511898} * ( ///
    growth_lag - {gearly=1.33764644398}*growth_early_lag ///
    - {grecent=1.34320513879}*growth_recent_lag - {gdata=0.599835853771}*growth_data_lag) ///
    + {gearly}*growth_early + {grecent}*growth_recent + {gdata}*growth_data ///
    + {gmarket=0.0194981964002}*market_centered), hasconstant(gc) vce(hac nwest 6) iterate(4000)
estimates store growth

* real_index
nl (real_index_target = {lc=29.1809587636} + {lar=0.748996244208} * ( ///
    real_index_lag - {learly=2.03137573694}*real_index_early_lag ///
    - {lrecent=2.88671476161}*real_index_recent_lag - {ldata=1.3019600765}*real_index_data_lag) ///
    + {learly}*real_index_early + {lrecent}*real_index_recent + {ldata}*real_index_data ///
    + {lmarket=-0.765088935896}*market_centered), hasconstant(lc) vce(hac nwest 6) iterate(4000)
estimates store real_index


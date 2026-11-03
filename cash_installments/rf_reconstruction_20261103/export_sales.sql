-- Read-only monthly residential aggregates; no credentials are stored here.
select period, count(*) as rows,
       sum(res_bought_rub) as gross_rub,
       sum(res_cancel_rub) as cancel_rub,
       sum(res_bought_m2) as gross_m2,
       sum(res_cancel_m2) as cancel_m2,
       count(*) filter (where res_bought_rub is null or res_cancel_rub is null
                        or res_bought_m2 is null or res_cancel_m2 is null) as null_rows
from prodoma.sales
group by period
order by period;


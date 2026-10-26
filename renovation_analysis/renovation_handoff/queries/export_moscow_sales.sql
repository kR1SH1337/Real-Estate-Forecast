-- Read-only query. Run in the analyst's own authorized PostgreSQL connection.
-- Export the result as UTF-8 CSV, separator ';', with column headers.
-- Query reconstructed for handoff; no connection settings or keys included.
SELECT building, project, developer, property_class, period::date AS period,
       res_bought_qty, res_bought_m2, res_bought_rub, res_cancel_m2
FROM prodoma.sales
WHERE region = 'Город Москва'
  AND period::date >= DATE '2021-01-01'
  AND period::date < DATE '2026-10-01'
ORDER BY building, period::date;

-- Denominators of the published working share are NET sales, not gross sales.
SELECT EXTRACT(YEAR FROM period)::integer AS year,
       SUM(net_m2) AS net_m2, SUM(net_rub) AS net_rub
FROM prodoma.sales_net
WHERE region = 'Город Москва'
  AND period >= DATE '2021-01-01'
  AND period < DATE '2026-10-01'
GROUP BY 1 ORDER BY 1;

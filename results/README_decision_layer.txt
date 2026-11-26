Decision layer run on modelling_table.csv (Li-first table B, charge balance <=10%, 4,639 rows with Li + lean Tier 1).

Scripts: decision_layer.py (models + out-of-fold predictions + integrity tests), metrics.py (tables), seed_check.py (conformal calibration-draw stability).
Seed 42. HistGradientBoosting: max_depth 3, learning_rate 0.05, max_iter 300, early_stopping off. Logistic: standardized, L2, C=1.0.

Schemes: V3 = leave-one-basin-out (basins with >=30 rows held out; smaller basins always in training). V4 = leave-one-IDDB-out (top 10 IDDBs).
Conformal (CQR, 80% nominal): calibration on whole groups. Candidate groups = training groups with >=30 rows; 25% of them (min 3) chosen with a fixed seed per fold; models for the interval fit on the remaining training groups.
Decision rule: Go if conformal lower bound >= log10(threshold); No-go if upper bound < log10(threshold); otherwise Measure-first. Thresholds 70 and 150 mg/L.

ASSUMPTIONS / DEVIATIONS
- Cl/Br ratio is a mass ratio; other ratios molar (a constant offset has no effect after scaling).
- ln(Ca/Mg) equals ln(Ca/Cl) - ln(Mg/Cl): one redundant feature, kept as specified.
- A single calibration draw per fold in the main run; seed_check.py repeats it with 5 other draws. Coverage and No-go counts move noticeably between draws.
- F1 and F2 rows differ (F2 needs K, Br, Sr), so F1-vs-F2 rows are NOT directly comparable in these tables.
- IDDB composition differs by 4 rows from an earlier table B (USGSBREIT 2,610 vs 2,614; Greater Green River 314 vs 315), probably dedupe tie-breaking. Totals are identical (4,639).
- False alarm = Go call on a row with true Li < 20 mg/L, for both thresholds.

INTEGRITY
T1 shuffled labels: pooled AUC 0.478-0.510 (expected ~0.5).
T2 no held-out group or row in training and no calibration group equal to the held-out group: asserted on all 51 folds.
T3 features exactly F1/F2, no identifiers, coordinates, TEMP or Li-derived columns: asserted.
T4 coverage is reported, not pass/fail (see results workbook).

IDENTICAL-ROWS COMPARISON (identical_rows.py, identical_metrics.py)
Same 2,230 rows (K, Br, Sr all present) for both feature sets. V3 = 12 held-out basins (2,087 rows); V4 = 10 held-out datasets (1,820 rows).
Calibration groups are the same for F1 and F2 in every fold; five calibration draws (seeds 1000*s+fold). Intervals/decisions are means over the five draws, with the range across draws.
95% intervals are percentile intervals from 1,000 bootstrap resamples of whole held-out groups (approximate with 10-12 groups).
Integrity: T1 shuffled-label AUCs 0.476-0.503; T2 asserted on 44 folds; T3 asserted.

NOTE ON LOGS
decision_layer_log.txt was emptied by a bug (importing decision_layer.py from later scripts reopened the log in write mode). The bug is fixed in decision_layer.py (log is now appended on demand). The real printouts of each run are saved as *_run_output.txt and decision_layer_metrics_output.txt.

SCREENING TOOL (build_screening_tool.py + screening_tool_template.html)
Final models trained on all rows: HistGradientBoosting classifier for Li >=70 mg/L plus, for the interval, five conformal draws (calibration on different held-out basins, seeds 1000+s); the page takes the median lower and upper bound across draws. F1 = routine ions (4,639 wells); F2 adds K, Br, Sr (2,230 wells; used only when all three are entered).
The exported trees were checked against scikit-learn on every training row, and the page's real JavaScript was run under Node on the nine Freeman samples (all No-go) and one Arkla training well (Go candidate; optimistic because it is training data).
No hold-out remains for these final models, so the tool's reliability statements come from the earlier leave-one-basin-out runs, not from the final fit.

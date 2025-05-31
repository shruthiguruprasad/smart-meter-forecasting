
📈 STEP 4: PROPHET FORECASTING
----------------------------------------
Running Prophet forecasting...
🚀 PROPHET DAY-AHEAD FORECASTING
========================================
📊 Available columns in dataframe: ['Acorn', 'Acorn_grouped', 'LCLid', 'acorn_avg_consumption', 'acorn_consumption_ratio', 'acorn_peak_ratio', 'acorn_variability', 'afternoon_kwh', 'base_load', 'base_load_ratio']...
   ✅ Added core regressor: temp_avg
   ✅ Added core regressor: heating_degree_days
   ✅ Added core regressor: cooling_degree_days
   ✅ Added core regressor: is_weekend
   ⚠️ Skipped humidity - low variance (0.010)
📊 Using 4 regressors to avoid overfitting
📊 Prophet data prepared: 701 days
📊 External regressors: ['temp_avg', 'heating_degree_days', 'cooling_degree_days', 'is_weekend']
[I 2025-05-31 06:51:12,074] A new study created in memory with name: no-name-a447aef6-d105-4b72-80e1-1482fc54d2b3
📊 Available columns in dataframe: ['Acorn', 'Acorn_grouped', 'LCLid', 'acorn_avg_consumption', 'acorn_consumption_ratio', 'acorn_peak_ratio', 'acorn_variability', 'afternoon_kwh', 'base_load', 'base_load_ratio']...
   ✅ Added core regressor: temp_avg
   ✅ Added core regressor: heating_degree_days
   ✅ Added core regressor: cooling_degree_days
   ✅ Added core regressor: is_weekend
   ⚠️ Skipped humidity - low variance (0.004)
📊 Using 4 regressors to avoid overfitting
📊 Prophet data prepared: 90 days
📊 External regressors: ['temp_avg', 'heating_degree_days', 'cooling_degree_days', 'is_weekend']

🎯 Starting hyperparameter tuning...
🎯 TUNING PROPHET HYPERPARAMETERS
========================================
Best trial: 8. Best value: 3.50007: 100%
 20/20 [00:15<00:00,  1.37it/s]
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:12 - cmdstanpy - INFO - Chain [1] done processing
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:12 - cmdstanpy - INFO - Chain [1] done processing
06:51:12 - cmdstanpy - INFO - Chain [1] done processing
06:51:12 - cmdstanpy - INFO - Chain [1] done processing
06:51:12 - cmdstanpy - INFO - Chain [1] done processing
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:12 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:12,667] Trial 0 finished with value: 3.9380721208861758 and parameters: {'changepoint_prior_scale': 0.010253509690168494, 'seasonality_prior_scale': 7.114476009343421, 'holidays_prior_scale': 1.5702970884055387, 'changepoint_range': 0.8897987726295555, 'n_changepoints': 18, 'seasonality_mode': 'additive'}. Best is trial 0 with value: 3.9380721208861758.
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:12 - cmdstanpy - INFO - Chain [1] start processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
06:51:13 - cmdstanpy - INFO - Chain [1] start processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:13,365] Trial 1 finished with value: 4.880757656471398 and parameters: {'changepoint_prior_scale': 0.21766241123453672, 'seasonality_prior_scale': 0.6358358856676253, 'holidays_prior_scale': 1.3311216080736887, 'changepoint_range': 0.8030876741443704, 'n_changepoints': 35, 'seasonality_mode': 'additive'}. Best is trial 0 with value: 3.9380721208861758.
06:51:13 - cmdstanpy - INFO - Chain [1] start processing
06:51:13 - cmdstanpy - INFO - Chain [1] start processing
06:51:13 - cmdstanpy - INFO - Chain [1] start processing
06:51:13 - cmdstanpy - INFO - Chain [1] start processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
06:51:13 - cmdstanpy - INFO - Chain [1] start processing
06:51:13 - cmdstanpy - INFO - Chain [1] done processing
06:51:14 - cmdstanpy - INFO - Chain [1] start processing
06:51:14 - cmdstanpy - INFO - Chain [1] start processing
[I 2025-05-31 06:51:13,904] Trial 2 finished with value: 3.5512294491963274 and parameters: {'changepoint_prior_scale': 0.003095566460242371, 'seasonality_prior_scale': 0.03549878832196503, 'holidays_prior_scale': 0.08179499475211674, 'changepoint_range': 0.8787134647448357, 'n_changepoints': 24, 'seasonality_mode': 'multiplicative'}. Best is trial 2 with value: 3.5512294491963274.
06:51:14 - cmdstanpy - INFO - Chain [1] start processing
06:51:14 - cmdstanpy - INFO - Chain [1] start processing
06:51:14 - cmdstanpy - INFO - Chain [1] done processing
06:51:14 - cmdstanpy - INFO - Chain [1] done processing
06:51:14 - cmdstanpy - INFO - Chain [1] done processing
06:51:14 - cmdstanpy - INFO - Chain [1] done processing
06:51:14 - cmdstanpy - INFO - Chain [1] start processing
06:51:14 - cmdstanpy - INFO - Chain [1] done processing
06:51:14 - cmdstanpy - INFO - Chain [1] start processing
[I 2025-05-31 06:51:14,443] Trial 3 finished with value: 3.708263089639061 and parameters: {'changepoint_prior_scale': 0.002379522116387727, 'seasonality_prior_scale': 0.07523742884534858, 'holidays_prior_scale': 0.1256277350380703, 'changepoint_range': 0.8684104976325554, 'n_changepoints': 31, 'seasonality_mode': 'multiplicative'}. Best is trial 2 with value: 3.5512294491963274.
06:51:14 - cmdstanpy - INFO - Chain [1] start processing
06:51:14 - cmdstanpy - INFO - Chain [1] start processing
06:51:14 - cmdstanpy - INFO - Chain [1] start processing
06:51:14 - cmdstanpy - INFO - Chain [1] done processing
06:51:14 - cmdstanpy - INFO - Chain [1] done processing
06:51:14 - cmdstanpy - INFO - Chain [1] done processing
06:51:14 - cmdstanpy - INFO - Chain [1] done processing
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:14,979] Trial 4 finished with value: 3.6673235880831228 and parameters: {'changepoint_prior_scale': 0.039710847107924725, 'seasonality_prior_scale': 0.013783237455007183, 'holidays_prior_scale': 0.6647135865318028, 'changepoint_range': 0.8255786185530938, 'n_changepoints': 16, 'seasonality_mode': 'multiplicative'}. Best is trial 2 with value: 3.5512294491963274.
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
[I 2025-05-31 06:51:15,617] Trial 5 finished with value: 4.814822350529904 and parameters: {'changepoint_prior_scale': 0.15199881220083966, 'seasonality_prior_scale': 0.08200518402245831, 'holidays_prior_scale': 0.019634341572933336, 'changepoint_range': 0.9026349539768235, 'n_changepoints': 24, 'seasonality_mode': 'multiplicative'}. Best is trial 2 with value: 3.5512294491963274.
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
06:51:15 - cmdstanpy - INFO - Chain [1] start processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
06:51:15 - cmdstanpy - INFO - Chain [1] done processing
06:51:16 - cmdstanpy - INFO - Chain [1] start processing
06:51:16 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:16,250] Trial 6 finished with value: 4.011888837247581 and parameters: {'changepoint_prior_scale': 0.0012382649697023553, 'seasonality_prior_scale': 5.345166110646819, 'holidays_prior_scale': 0.059750279999602945, 'changepoint_range': 0.8993783426530972, 'n_changepoints': 21, 'seasonality_mode': 'multiplicative'}. Best is trial 2 with value: 3.5512294491963274.
06:51:16 - cmdstanpy - INFO - Chain [1] start processing
06:51:16 - cmdstanpy - INFO - Chain [1] start processing
06:51:16 - cmdstanpy - INFO - Chain [1] start processing
06:51:16 - cmdstanpy - INFO - Chain [1] start processing
06:51:16 - cmdstanpy - INFO - Chain [1] done processing
06:51:16 - cmdstanpy - INFO - Chain [1] done processing
06:51:16 - cmdstanpy - INFO - Chain [1] done processing
06:51:16 - cmdstanpy - INFO - Chain [1] done processing
06:51:16 - cmdstanpy - INFO - Chain [1] start processing
06:51:16 - cmdstanpy - INFO - Chain [1] done processing
06:51:16 - cmdstanpy - INFO - Chain [1] start processing
[I 2025-05-31 06:51:16,808] Trial 7 finished with value: 3.6099245358957583 and parameters: {'changepoint_prior_scale': 0.0031543990308330965, 'seasonality_prior_scale': 8.10501612641158, 'holidays_prior_scale': 2.1154290797261215, 'changepoint_range': 0.9409248412346283, 'n_changepoints': 33, 'seasonality_mode': 'multiplicative'}. Best is trial 2 with value: 3.5512294491963274.
06:51:17 - cmdstanpy - INFO - Chain [1] start processing
06:51:17 - cmdstanpy - INFO - Chain [1] start processing
06:51:17 - cmdstanpy - INFO - Chain [1] start processing
06:51:17 - cmdstanpy - INFO - Chain [1] done processing
06:51:17 - cmdstanpy - INFO - Chain [1] done processing
06:51:17 - cmdstanpy - INFO - Chain [1] done processing
06:51:17 - cmdstanpy - INFO - Chain [1] done processing
06:51:17 - cmdstanpy - INFO - Chain [1] start processing
06:51:17 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:17,373] Trial 8 finished with value: 3.5000679582320013 and parameters: {'changepoint_prior_scale': 0.0017331598058558703, 'seasonality_prior_scale': 0.03872118032174583, 'holidays_prior_scale': 0.013667272915456224, 'changepoint_range': 0.8487995496144897, 'n_changepoints': 23, 'seasonality_mode': 'multiplicative'}. Best is trial 8 with value: 3.5000679582320013.
06:51:17 - cmdstanpy - INFO - Chain [1] start processing
06:51:17 - cmdstanpy - INFO - Chain [1] start processing
06:51:17 - cmdstanpy - INFO - Chain [1] start processing
06:51:17 - cmdstanpy - INFO - Chain [1] start processing
06:51:17 - cmdstanpy - INFO - Chain [1] done processing
06:51:17 - cmdstanpy - INFO - Chain [1] done processing
06:51:17 - cmdstanpy - INFO - Chain [1] done processing
06:51:17 - cmdstanpy - INFO - Chain [1] done processing
06:51:17 - cmdstanpy - INFO - Chain [1] start processing
06:51:18 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:17,934] Trial 9 finished with value: 3.602111328407133 and parameters: {'changepoint_prior_scale': 0.00918050407457243, 'seasonality_prior_scale': 0.06963114377829285, 'holidays_prior_scale': 0.4247058562261869, 'changepoint_range': 0.8211386337462144, 'n_changepoints': 31, 'seasonality_mode': 'multiplicative'}. Best is trial 8 with value: 3.5000679582320013.
06:51:18 - cmdstanpy - INFO - Chain [1] start processing
06:51:18 - cmdstanpy - INFO - Chain [1] start processing
06:51:18 - cmdstanpy - INFO - Chain [1] start processing
06:51:18 - cmdstanpy - INFO - Chain [1] start processing
06:51:18 - cmdstanpy - INFO - Chain [1] done processing
06:51:18 - cmdstanpy - INFO - Chain [1] done processing
06:51:18 - cmdstanpy - INFO - Chain [1] done processing
06:51:18 - cmdstanpy - INFO - Chain [1] done processing
06:51:18 - cmdstanpy - INFO - Chain [1] start processing
06:51:18 - cmdstanpy - INFO - Chain [1] done processing
06:51:18 - cmdstanpy - ERROR - Chain [1] error: error during processing Operation not permitted
06:51:18 - cmdstanpy - INFO - Chain [1] start processing
[I 2025-05-31 06:51:18,538] Trial 10 finished with value: 3.903521641472614 and parameters: {'changepoint_prior_scale': 0.02434232144332388, 'seasonality_prior_scale': 0.467553113578448, 'holidays_prior_scale': 8.102356207766633, 'changepoint_range': 0.8520276337109933, 'n_changepoints': 28, 'seasonality_mode': 'additive'}. Best is trial 8 with value: 3.5000679582320013.
06:51:19 - cmdstanpy - INFO - Chain [1] done processing
06:51:19 - cmdstanpy - INFO - Chain [1] start processing
06:51:19 - cmdstanpy - INFO - Chain [1] start processing
06:51:19 - cmdstanpy - INFO - Chain [1] start processing
06:51:19 - cmdstanpy - INFO - Chain [1] start processing
06:51:19 - cmdstanpy - INFO - Chain [1] done processing
06:51:19 - cmdstanpy - ERROR - Chain [1] error: error during processing Operation not permitted
06:51:19 - cmdstanpy - INFO - Chain [1] done processing
06:51:19 - cmdstanpy - ERROR - Chain [1] error: error during processing Operation not permitted
06:51:19 - cmdstanpy - INFO - Chain [1] start processing
06:51:19 - cmdstanpy - INFO - Chain [1] done processing
06:51:19 - cmdstanpy - INFO - Chain [1] done processing
06:51:19 - cmdstanpy - INFO - Chain [1] start processing
06:51:20 - cmdstanpy - INFO - Chain [1] done processing
06:51:20 - cmdstanpy - INFO - Chain [1] done processing
06:51:20 - cmdstanpy - INFO - Chain [1] start processing
06:51:20 - cmdstanpy - INFO - Chain [1] done processing
06:51:20 - cmdstanpy - ERROR - Chain [1] error: error during processing Operation not permitted
06:51:20 - cmdstanpy - INFO - Chain [1] start processing
[I 2025-05-31 06:51:20,482] Trial 11 finished with value: 3.52891893369554 and parameters: {'changepoint_prior_scale': 0.001101767345549512, 'seasonality_prior_scale': 0.010811131532236833, 'holidays_prior_scale': 0.01442840747886248, 'changepoint_range': 0.8551591576174836, 'n_changepoints': 24, 'seasonality_mode': 'multiplicative'}. Best is trial 8 with value: 3.5000679582320013.
06:51:21 - cmdstanpy - INFO - Chain [1] done processing
06:51:21 - cmdstanpy - INFO - Chain [1] start processing
06:51:21 - cmdstanpy - INFO - Chain [1] start processing
06:51:21 - cmdstanpy - INFO - Chain [1] start processing
06:51:21 - cmdstanpy - INFO - Chain [1] start processing
06:51:21 - cmdstanpy - INFO - Chain [1] done processing
06:51:21 - cmdstanpy - ERROR - Chain [1] error: error during processing Operation not permitted
06:51:21 - cmdstanpy - INFO - Chain [1] done processing
06:51:21 - cmdstanpy - ERROR - Chain [1] error: error during processing Operation not permitted
06:51:21 - cmdstanpy - INFO - Chain [1] start processing
06:51:21 - cmdstanpy - INFO - Chain [1] done processing
06:51:21 - cmdstanpy - ERROR - Chain [1] error: error during processing Operation not permitted
06:51:21 - cmdstanpy - INFO - Chain [1] done processing
06:51:21 - cmdstanpy - INFO - Chain [1] start processing
06:51:21 - cmdstanpy - INFO - Chain [1] start processing
06:51:22 - cmdstanpy - INFO - Chain [1] done processing
06:51:22 - cmdstanpy - INFO - Chain [1] done processing
06:51:22 - cmdstanpy - INFO - Chain [1] done processing
06:51:22 - cmdstanpy - INFO - Chain [1] start processing
06:51:22 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:22,800] Trial 12 finished with value: 3.955357291041557 and parameters: {'changepoint_prior_scale': 0.001157923793436067, 'seasonality_prior_scale': 0.012153210612944224, 'holidays_prior_scale': 0.012379205103725864, 'changepoint_range': 0.8477429375187212, 'n_changepoints': 27, 'seasonality_mode': 'multiplicative'}. Best is trial 8 with value: 3.5000679582320013.
06:51:23 - cmdstanpy - INFO - Chain [1] start processing
06:51:23 - cmdstanpy - INFO - Chain [1] start processing
06:51:23 - cmdstanpy - INFO - Chain [1] start processing
06:51:23 - cmdstanpy - INFO - Chain [1] start processing
06:51:23 - cmdstanpy - INFO - Chain [1] done processing
06:51:23 - cmdstanpy - INFO - Chain [1] done processing
06:51:23 - cmdstanpy - INFO - Chain [1] done processing
06:51:23 - cmdstanpy - INFO - Chain [1] done processing
06:51:23 - cmdstanpy - INFO - Chain [1] start processing
06:51:23 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:23,371] Trial 13 finished with value: 3.5335332806015596 and parameters: {'changepoint_prior_scale': 0.009588711865887266, 'seasonality_prior_scale': 0.027056636258998464, 'holidays_prior_scale': 0.03080379033945705, 'changepoint_range': 0.8487713800898555, 'n_changepoints': 20, 'seasonality_mode': 'multiplicative'}. Best is trial 8 with value: 3.5000679582320013.
06:51:23 - cmdstanpy - INFO - Chain [1] start processing
06:51:23 - cmdstanpy - INFO - Chain [1] start processing
06:51:23 - cmdstanpy - INFO - Chain [1] start processing
06:51:23 - cmdstanpy - INFO - Chain [1] start processing
06:51:23 - cmdstanpy - INFO - Chain [1] done processing
06:51:23 - cmdstanpy - INFO - Chain [1] done processing
06:51:23 - cmdstanpy - INFO - Chain [1] done processing
06:51:23 - cmdstanpy - INFO - Chain [1] done processing
06:51:24 - cmdstanpy - INFO - Chain [1] start processing
06:51:24 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:24,044] Trial 14 finished with value: 4.940256438560918 and parameters: {'changepoint_prior_scale': 0.07805789099500472, 'seasonality_prior_scale': 0.18561860486787587, 'holidays_prior_scale': 0.010411689173589323, 'changepoint_range': 0.9221698491776741, 'n_changepoints': 22, 'seasonality_mode': 'additive'}. Best is trial 8 with value: 3.5000679582320013.
06:51:24 - cmdstanpy - INFO - Chain [1] start processing
06:51:24 - cmdstanpy - INFO - Chain [1] start processing
06:51:24 - cmdstanpy - INFO - Chain [1] start processing
06:51:24 - cmdstanpy - INFO - Chain [1] start processing
06:51:24 - cmdstanpy - INFO - Chain [1] done processing
06:51:24 - cmdstanpy - INFO - Chain [1] done processing
06:51:24 - cmdstanpy - INFO - Chain [1] done processing
06:51:24 - cmdstanpy - INFO - Chain [1] done processing
06:51:24 - cmdstanpy - INFO - Chain [1] start processing
06:51:24 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:24,828] Trial 15 finished with value: 4.258956256687546 and parameters: {'changepoint_prior_scale': 0.49802282257615405, 'seasonality_prior_scale': 0.01017116084587649, 'holidays_prior_scale': 0.20890673800818002, 'changepoint_range': 0.8613498161264435, 'n_changepoints': 27, 'seasonality_mode': 'multiplicative'}. Best is trial 8 with value: 3.5000679582320013.
06:51:25 - cmdstanpy - INFO - Chain [1] start processing
06:51:25 - cmdstanpy - INFO - Chain [1] start processing
06:51:25 - cmdstanpy - INFO - Chain [1] start processing
06:51:25 - cmdstanpy - INFO - Chain [1] start processing
06:51:25 - cmdstanpy - INFO - Chain [1] done processing
06:51:25 - cmdstanpy - INFO - Chain [1] done processing
06:51:25 - cmdstanpy - INFO - Chain [1] done processing
06:51:25 - cmdstanpy - INFO - Chain [1] done processing
06:51:25 - cmdstanpy - INFO - Chain [1] start processing
06:51:25 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:25,405] Trial 16 finished with value: 3.520157156924369 and parameters: {'changepoint_prior_scale': 0.0056326449971168275, 'seasonality_prior_scale': 1.9234357845908685, 'holidays_prior_scale': 0.035956649622782146, 'changepoint_range': 0.8315644142208012, 'n_changepoints': 23, 'seasonality_mode': 'multiplicative'}. Best is trial 8 with value: 3.5000679582320013.
06:51:25 - cmdstanpy - INFO - Chain [1] start processing
06:51:25 - cmdstanpy - INFO - Chain [1] start processing
06:51:25 - cmdstanpy - INFO - Chain [1] start processing
06:51:25 - cmdstanpy - INFO - Chain [1] start processing
06:51:25 - cmdstanpy - INFO - Chain [1] done processing
06:51:25 - cmdstanpy - INFO - Chain [1] done processing
06:51:25 - cmdstanpy - INFO - Chain [1] done processing
06:51:25 - cmdstanpy - INFO - Chain [1] done processing
06:51:26 - cmdstanpy - INFO - Chain [1] start processing
06:51:26 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:26,014] Trial 17 finished with value: 3.5345820600531774 and parameters: {'changepoint_prior_scale': 0.00640307395605393, 'seasonality_prior_scale': 1.4485396290488697, 'holidays_prior_scale': 0.03976052548400756, 'changepoint_range': 0.8342074225577611, 'n_changepoints': 18, 'seasonality_mode': 'multiplicative'}. Best is trial 8 with value: 3.5000679582320013.
06:51:26 - cmdstanpy - INFO - Chain [1] start processing
06:51:26 - cmdstanpy - INFO - Chain [1] start processing
06:51:26 - cmdstanpy - INFO - Chain [1] start processing
06:51:26 - cmdstanpy - INFO - Chain [1] start processing
06:51:26 - cmdstanpy - INFO - Chain [1] done processing
06:51:26 - cmdstanpy - INFO - Chain [1] done processing
06:51:26 - cmdstanpy - INFO - Chain [1] done processing
06:51:26 - cmdstanpy - INFO - Chain [1] done processing
06:51:26 - cmdstanpy - INFO - Chain [1] start processing
06:51:26 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:26,736] Trial 18 finished with value: 3.5604026110334748 and parameters: {'changepoint_prior_scale': 0.004871092823853089, 'seasonality_prior_scale': 2.203514600082713, 'holidays_prior_scale': 0.030553420170511107, 'changepoint_range': 0.8118706965099795, 'n_changepoints': 15, 'seasonality_mode': 'additive'}. Best is trial 8 with value: 3.5000679582320013.
06:51:27 - cmdstanpy - INFO - Chain [1] start processing
06:51:27 - cmdstanpy - INFO - Chain [1] start processing
06:51:27 - cmdstanpy - INFO - Chain [1] start processing
06:51:27 - cmdstanpy - INFO - Chain [1] start processing
06:51:27 - cmdstanpy - INFO - Chain [1] done processing
06:51:27 - cmdstanpy - INFO - Chain [1] done processing
06:51:27 - cmdstanpy - INFO - Chain [1] done processing
06:51:27 - cmdstanpy - INFO - Chain [1] done processing
06:51:27 - cmdstanpy - INFO - Chain [1] start processing
06:51:27 - cmdstanpy - INFO - Chain [1] done processing
[I 2025-05-31 06:51:27,411] Trial 19 finished with value: 3.7226814456066655 and parameters: {'changepoint_prior_scale': 0.020030270544088947, 'seasonality_prior_scale': 0.23269635005832504, 'holidays_prior_scale': 0.10047109499401811, 'changepoint_range': 0.8337195104508982, 'n_changepoints': 22, 'seasonality_mode': 'multiplicative'}. Best is trial 8 with value: 3.5000679582320013.

📊 Best Prophet Hyperparameters:
   changepoint_prior_scale: 0.0017331598058558703
   seasonality_prior_scale: 0.03872118032174583
   holidays_prior_scale: 0.013667272915456224
   changepoint_range: 0.8487995496144897
   n_changepoints: 23
   seasonality_mode: multiplicative
   interval_width: 0.95
   mcmc_samples: 0
   uncertainty_samples: 1000

🎯 Best Cross-Validation MAE: 3.5001
✅ Hyperparameter tuning completed

✅ Using tuned hyperparameters
✅ Prophet model created with changepoint_prior=0.002, seasonality_prior=0.039
🚀 Training Prophet model...
📊 Added regressor: temp_avg
📊 Added regressor: heating_degree_days
📊 Added regressor: cooling_degree_days
📊 Added regressor: is_weekend
✅ Prophet model trained successfully
📊 Found 152 missing values in date gaps - filling with interpolation
📊 Filled 38 gaps in temp_avg
📊 Filled 38 gaps in heating_degree_days
📊 Filled 38 gaps in cooling_degree_days
📊 Filled 38 gaps in is_weekend
📊 Combined regressors from training and test data
📊 Training period: 2011-11-24 00:00:00 to 2013-10-29 00:00:00
📊 Test period: 2013-11-29 00:00:00 to 2014-02-27 00:00:00
📊 Combined regressor range: 2011-11-24 00:00:00 to 2014-02-27 00:00:00
📊 Total days covered: 827
📊 Processed regressor temp_avg: NaN count = 0
📊 Processed regressor heating_degree_days: NaN count = 0
📊 Processed regressor cooling_degree_days: NaN count = 0
📊 Processed regressor is_weekend: NaN count = 0
📊 Future regressors prepared from combined data: (827, 5)
📊 Regressor columns: ['temp_avg', 'heating_degree_days', 'cooling_degree_days', 'is_weekend']
📊 Any remaining NaNs in future regressors: 0
📊 Any remaining NaNs in training data: 0
🔮 Generating 90-day forecast...
📊 Adding future regressors...
📊 Future regressors shape: (827, 5)
📊 Future regressors NaN count: 0
📊 Future regressors date range: 2011-11-24 00:00:00 to 2014-02-27 00:00:00
✅ temp_avg: no NaNs after merge
✅ heating_degree_days: no NaNs after merge
✅ cooling_degree_days: no NaNs after merge
✅ is_weekend: no NaNs after merge
✅ Successfully merged all 4 regressors
✅ Final check: temp_avg has no NaNs in forecast period
✅ Final check: heating_degree_days has no NaNs in forecast period
✅ Final check: cooling_degree_days has no NaNs in forecast period
✅ Final check: is_weekend has no NaNs in forecast period
✅ Forecast generated for 90 days

⚠️  Poor performance detected - running diagnostics...
🔍 PROPHET PERFORMANCE DIAGNOSTICS
========================================
1️⃣ ALIGNMENT CHECK:
   y_true length: 90
   y_pred length: 90
   dates length: 90
   First few actual vs predicted:
   Date 2013-11-29T00:00:00.000000000: Actual=33.71, Pred=32.38
   Date 2013-11-30T00:00:00.000000000: Actual=36.90, Pred=33.28
   Date 2013-12-01T00:00:00.000000000: Actual=45.26, Pred=33.53
   Date 2013-12-02T00:00:00.000000000: Actual=40.82, Pred=34.45
   Date 2013-12-03T00:00:00.000000000: Actual=45.40, Pred=37.28

2️⃣ DATA VOLATILITY CHECK:
   Training: Mean=30.17, Std=12.38, CV=0.411
   Test: Mean=34.61, Std=6.01, CV=0.411
   Volatility assessment: LOW

3️⃣ BENCHMARK COMPARISONS:
   Naive (mean): MAE=5.78, R²=-0.548
   Last value: MAE=23.79
   7-day rolling: MAE=17.89
   Prophet: MAE=11.31, R²=-3.668

4️⃣ HETEROSKEDASTICITY CHECK:
   Correlation(pred, |residuals|): 0.682 (p=0.000)
   Heteroskedasticity: YES - try log transform

5️⃣ SERIES CHARACTERISTICS:
   Weekly autocorrelation: 0.353
   Trend: WEAK

6️⃣ RECOMMENDATIONS:
   ❌ Prophet worse than naive - try simpler model
   📊 Try log transformation for heteroskedasticity
✅ Prophet forecasting completed
✅ Prophet forecasting completed
📊 Forecast summary:
   Test period: 90 days
   Actual range: 22.4 - 54.0 kWh
   Predicted range: 32.4 - 52.2 kWh


📊 STEP 5: MODEL EVALUATION
----------------------------------------

📊 Prophet Forecast Evaluation:
   MAE:  11.312 kWh
   RMSE: 12.979 kWh
   MAPE: 36.10%
   R²:   -3.668
📋 PROPHET PERFORMANCE METRICS:
   MAE:  11.312 kWh
   RMSE: 12.979 kWh
   MAPE: 36.10%
   R²:   -3.668
   Overall: 🔴 POOR - Worse than naive baseline

📊 Additional insights:
   Mean residual: -9.341 kWh
   Residual std: 9.011 kWh
   Max overestimate: 14.8 kWh
   Max underestimate: -27.6 kWh

   🔄 STEP 6: BASELINE COMPARISON
----------------------------------------
Running Prophet with default parameters for comparison...
🚀 PROPHET DAY-AHEAD FORECASTING
========================================
📊 Available columns in dataframe: ['Acorn', 'Acorn_grouped', 'LCLid', 'acorn_avg_consumption', 'acorn_consumption_ratio', 'acorn_peak_ratio', 'acorn_variability', 'afternoon_kwh', 'base_load', 'base_load_ratio']...
   ✅ Added core regressor: temp_avg
   ✅ Added core regressor: heating_degree_days
   ✅ Added core regressor: cooling_degree_days
   ✅ Added core regressor: is_weekend
   ⚠️ Skipped humidity - low variance (0.010)
📊 Using 4 regressors to avoid overfitting
📊 Prophet data prepared: 701 days
📊 External regressors: ['temp_avg', 'heating_degree_days', 'cooling_degree_days', 'is_weekend']
06:51:49 - cmdstanpy - INFO - Chain [1] start processing
06:51:49 - cmdstanpy - INFO - Chain [1] done processing
📊 Available columns in dataframe: ['Acorn', 'Acorn_grouped', 'LCLid', 'acorn_avg_consumption', 'acorn_consumption_ratio', 'acorn_peak_ratio', 'acorn_variability', 'afternoon_kwh', 'base_load', 'base_load_ratio']...
   ✅ Added core regressor: temp_avg
   ✅ Added core regressor: heating_degree_days
   ✅ Added core regressor: cooling_degree_days
   ✅ Added core regressor: is_weekend
   ⚠️ Skipped humidity - low variance (0.004)
📊 Using 4 regressors to avoid overfitting
📊 Prophet data prepared: 90 days
📊 External regressors: ['temp_avg', 'heating_degree_days', 'cooling_degree_days', 'is_weekend']
⚠️  Using default hyperparameters (consider tuning for better performance)
✅ Prophet model created with changepoint_prior=0.050, seasonality_prior=1.000
🚀 Training Prophet model...
📊 Added regressor: temp_avg
📊 Added regressor: heating_degree_days
📊 Added regressor: cooling_degree_days
📊 Added regressor: is_weekend
✅ Prophet model trained successfully
📊 Found 152 missing values in date gaps - filling with interpolation
📊 Filled 38 gaps in temp_avg
📊 Filled 38 gaps in heating_degree_days
📊 Filled 38 gaps in cooling_degree_days
📊 Filled 38 gaps in is_weekend
📊 Combined regressors from training and test data
📊 Training period: 2011-11-24 00:00:00 to 2013-10-29 00:00:00
📊 Test period: 2013-11-29 00:00:00 to 2014-02-27 00:00:00
📊 Combined regressor range: 2011-11-24 00:00:00 to 2014-02-27 00:00:00
📊 Total days covered: 827
📊 Processed regressor temp_avg: NaN count = 0
📊 Processed regressor heating_degree_days: NaN count = 0
📊 Processed regressor cooling_degree_days: NaN count = 0
📊 Processed regressor is_weekend: NaN count = 0
📊 Future regressors prepared from combined data: (827, 5)
📊 Regressor columns: ['temp_avg', 'heating_degree_days', 'cooling_degree_days', 'is_weekend']
📊 Any remaining NaNs in future regressors: 0
📊 Any remaining NaNs in training data: 0
🔮 Generating 90-day forecast...
📊 Adding future regressors...
📊 Future regressors shape: (827, 5)
📊 Future regressors NaN count: 0
📊 Future regressors date range: 2011-11-24 00:00:00 to 2014-02-27 00:00:00
✅ temp_avg: no NaNs after merge
✅ heating_degree_days: no NaNs after merge
✅ cooling_degree_days: no NaNs after merge
✅ is_weekend: no NaNs after merge
✅ Successfully merged all 4 regressors
✅ Final check: temp_avg has no NaNs in forecast period
✅ Final check: heating_degree_days has no NaNs in forecast period
✅ Final check: cooling_degree_days has no NaNs in forecast period
✅ Final check: is_weekend has no NaNs in forecast period
✅ Forecast generated for 90 days

⚠️  Poor performance detected - running diagnostics...
🔍 PROPHET PERFORMANCE DIAGNOSTICS
========================================
1️⃣ ALIGNMENT CHECK:
   y_true length: 90
   y_pred length: 90
   dates length: 90
   First few actual vs predicted:
   Date 2013-11-29T00:00:00.000000000: Actual=33.71, Pred=28.73
   Date 2013-11-30T00:00:00.000000000: Actual=36.90, Pred=29.36
   Date 2013-12-01T00:00:00.000000000: Actual=45.26, Pred=29.41
   Date 2013-12-02T00:00:00.000000000: Actual=40.82, Pred=29.84
   Date 2013-12-03T00:00:00.000000000: Actual=45.40, Pred=32.53

2️⃣ DATA VOLATILITY CHECK:
   Training: Mean=30.17, Std=12.38, CV=0.411
   Test: Mean=34.61, Std=6.01, CV=0.411
   Volatility assessment: LOW

3️⃣ BENCHMARK COMPARISONS:
   Naive (mean): MAE=5.78, R²=-0.548
   Last value: MAE=23.79
   7-day rolling: MAE=17.89
   Prophet: MAE=6.96, R²=-1.077

4️⃣ HETEROSKEDASTICITY CHECK:
   Correlation(pred, |residuals|): 0.210 (p=0.047)
   Heteroskedasticity: NO

5️⃣ SERIES CHARACTERISTICS:
   Weekly autocorrelation: 0.353
   Trend: WEAK

6️⃣ RECOMMENDATIONS:
   ❌ Prophet worse than naive - try simpler model

💡 TIP: Consider setting tune_hyperparameters=True to improve performance
✅ Prophet forecasting completed

📊 Prophet (Default) Forecast Evaluation:
   MAE:  6.963 kWh
   RMSE: 8.657 kWh
   MAPE: 21.48%
   R²:   -1.077

📊 TUNED vs DEFAULT COMPARISON:
                  Tuned    Default   Improvement
   MAE:          11.312    6.963     -62.5%
   RMSE:         12.979    8.657     -49.9%
   MAPE:         36.10%   21.48%    -68.1%
   R²:           -3.668    -1.077     -240.6%

📊 SIMPLE BASELINE COMPARISONS:
   Naive (mean):     MAE=5.772, R²=-0.546
   Prophet:          MAE=11.312, R²=-3.668
   Improvement:      -96.0% MAE, -571.5% R²
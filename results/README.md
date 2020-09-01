## Results

This folder contains a list of results;

- boxplot_features_imp_rf_size details the feature importances for the encdoed sizes of input videos

- boxplot_features_imp_rf_kbs is the **figure 2a**.

- boxplot_features_imp_linear_kbs is the **figure 2b**.

- boxplot_imp_group0-3 depicts the boxplots of feature importances for encoded sizes for the four groups of performances (similar to those constructed with bitrates)

- config_min_std_ranking (and max) depicts the rankings for the configurations having the least and the most input sensitive profile (biggest standard deviations)

- config_tune is the result of an attempt of "tune table", that did not work as well as Inputec - each line is the best configuration chosen by the tune table. But the ratios between this tune table and the average values were about 0.88, and were outperformed by Inputec.

- corr_distrib is (maybe an old version of) the summary of the correlation distribution (relative to the first figure)

- corr_inter_group is a high level correlogram between groups of performances (for encoding sizes)

- lots of corr_matrix_* files depict correlogram, for different metrics (Spearman, Pearson, Kullback Leibler divergence) and performances

- corrmatrix_kbs_modif if **figure 1**.

- encoding_profiles detail the differences between performance groups

- Inputec approach is the high level description of the approach, and correponds to the **figure 3**.

- list_feature_importances_* are features importances computed input per input, to have a low level description of importances. Several performances lead to several files.

- original * are measurements

- res_box_baseline and res_box_baseline_rank are the **figures 5a and 5b**

- shortlist_importances_* is an excerpt of feature importances. Influential features are highlighted in green. The related excel : tab_imp_rf 

- tab_group is an excel file describing the different performance groups

- tiny_tab_group is the short version of tab_group, to fit in the paper. It represents the **figure 4**.

- transition_table is an explanation of the old approach we were suggesting. But it was outperformed by Inputec.


# ED
Scripts for Machine Learning on Predictors of Eating Disorder Outcomes

This is a companion repository to our report of the most predictive factors for self-reported BMI after discharge from intensive services. It does not contain data which can be requested from the first author, but does contain all commands to disambiguate settings and algorithms described in the report.

The final script diverged from the sample in that the train and test sets were imputed separately and not a single imputation over the whole data before split. Additionally, the same full or deficient rank encoding was enforced prior to _caret_ training as described in the supplement.

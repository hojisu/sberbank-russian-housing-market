
import statsmodels.api as sm
import numpy as np

def by_f_test(df, formula, repeat=10, log_dv = True):
    result = None
    selected_ivs = []
    for i in range(repeat):
        model = sm.OLS.from_formula(formula, data=df)
        result = model.fit()
        anova = sm.stats.anova_lm(result, typ=2)
        selected_ivs = [iv[0] for iv in anova.iterrows() if iv[1][3] < 0.05]
        if len(selected_ivs) >= 0:
            if log_dv == True:  
                formula = 'np.log(_price_doc) ~ ' + ' + '.join(selected_ivs)
            else:
                formula = '_price_doc ~ ' + ' + '.join(selected_ivs)
        else:
            return result, selected_ivs
    return result, selected_ivs,  formula

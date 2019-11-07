
import statsmodels as sm

def make_statsmodels_ols_formula(numeric_ivs, categorical_ivs, dv, log_vs=[], degree=1, scale=True):


    if len(log_vs) > 0:
        numeric_ivs = ["np.log({})".format(iv) if iv in log_vs else iv for iv in numeric_ivs ]

    polynomials = []
    if degree > 1:
        for i in range(2, degree + 1):
            if scale:
                polynomials = list(map(lambda iv: 'scale(I({}**{}))'.format(iv, i), numeric_ivs))
            else:
                polynomials = list(map(lambda iv: 'I({}**{})'.format(iv, i), numeric_ivs))
    
    if scale:
        numeric_ivs = ["scale({})".format(iv) if scale else iv for iv in numeric_ivs ]

    formula = ''
    if dv in log_vs:
        formula = 'np.log({}) ~ '.format(dv)
    else:
        formula = '{} ~ '.format(dv)
    

    if len(categorical_ivs) > 0:
        if len(numeric_ivs) > 0:
            formula += " + ".join(list(map(lambda iv: 'C({})'.format(iv), categorical_ivs)))
        else:
            formula += " + ".join(list(map(lambda iv: 'C({})-1'.format(iv), categorical_ivs)))
    
    if len(polynomials) > 0:
        if len(categorical_ivs) > 0:
            return  formula + " + " + " + ".join(numeric_ivs) + " + " + " + ".join(polynomials)
        else:
            return  formula + " + ".join(numeric_ivs) + " + " + " + ".join(polynomials)
    else:
        if len(categorical_ivs) > 0:
            return formula + " + " + " + ".join(numeric_ivs)
        else:
            return formula + " + ".join(numeric_ivs) 

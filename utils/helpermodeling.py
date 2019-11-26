
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.metrics import r2_score, explained_variance_score



def regression(regressor, x_train, x_test, y_train):
    reg = regressor
    reg.fit(x_train, y_train)
    
    y_train_reg = reg.predict(x_train)
    y_test_reg = reg.predict(x_test)
    
    return y_train_reg, y_test_reg

def loss_plot(fit_history):
    plt.figure(figsize=(18,4))
    
    plt.plot(fit_history.history['loss'], color='#348ABD', label = 'train')
    plt.plot(fit_history.history['val_loss'], color='#228B22', label = 'test')
    plt.legend()
    plt.title('Loss Function')
    
def mae_plot(fit_history):
    plt.figure(figsize=(18, 4))
    
    plt.plot(fit_history.history['mean_absolute_error'], color = '#348ABD', label = 'train')
    plt.plot(fit_history.history['val_mean_absolute_error'], color='#228B22', label = 'test')
    plt.legend()
    plt.title('Mean Absolute Error')
    
def scores(regressor, y_train, y_test, y_train_reg, y_test_reg):
    print("______________________________________________")
    print(regressor)
    print("______________________________________________")
    print("R2 score. Train: ", r2_score(y_train, y_train_reg))
    print("R2 score. Test: ", r2_score(y_test, y_test_reg))
    print("______________________________________________")
    print("MSE score. Train: ", mean_squared_error(y_train, y_train_reg))
    print("MSE score. Test: ", mean_squared_error(y_test, y_test_reg))
    print("______________________________________________")
    print("RMSE score. Train: ", np.sqrt(mean_squared_error(y_train, y_train_reg)))
    print("RMSE score. Test: ", np.sqrt(mean_squared_error(y_train, y_train_reg)))
    print("______________________________________________")
    print("MAE score. Train: ", mean_absolute_error(y_train, y_train_reg))
    print("MAE score. Test: ", mean_absolute_error(y_test, y_test_reg))
    
def scores2(regressor, target, target_predict):
    print("______________________________________________")
    print(regressor)
    print("______________________________________________")
    print('R2 score:', r2_score(target, target_predict))
    print("______________________________________________")
    print('MSE socre:', mean_squared_error(target, target_predict))
    print("______________________________________________")
    print('RMSE socre:', np.sqrt(mean_squared_error(y_train, y_train_reg)))
    print("______________________________________________")
    print('MAE score:', mean_absolute_error(target, target_predict))

import sys
sys.path.append('.')
import joblib
from toolbox import *
from alphalens.performance import mean_return_by_quantile
import alphalens
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

FILE_NAME='lstm'
EPOCHS=3000
BATCH_SIZE=300
LR=0.01
LENS=10
MODEL=SimpleLSTM(input_size=2,hidden_size=20).double()
LOSS=Lloss()

prices=joblib.load('output/main_freq_price_df.m')
term_fac_df=joblib.load('output/pca_fac_including_all_code_df.m')[['PCA2','PCA3']]

prices_train=prices.loc[:'2020-01-01',:]
term_fac_df_train=term_fac_df.loc[:'2020-01-01',:]


data_X=get_X_seq(term_fac_df_train,LENS)
data_y=get_y(prices_train,2) # retain a day to turnover


X_train=torch.from_numpy(data_X)
y_train=torch.from_numpy(data_y)




data_X=get_X_seq(term_fac_df,LENS)
data_y=get_y(prices,2) # retain a day to turnover


X=torch.from_numpy(data_X)
y=torch.from_numpy(data_y)




#joblib.dump(value=X, filename='output/X.m')
#joblib.dump(value=y, filename='output/y.m')


if not os.path.exists(f'output/{FILE_NAME}'):
    os.makedirs(f'output/{FILE_NAME}')



model,running_loss=train(X_train,y_train,MODEL,LOSS,BATCH_SIZE,EPOCHS)
#joblib.dump(value=model, filename=f'output/{FILE_NAME}/model.m')
#joblib.dump(value=model, filename=f'output/{FILE_NAME}/loss_list.m')


score=model(X).flatten().detach().numpy()
learn2rank_fac=pd.DataFrame(score,index=term_fac_df.index,columns=['learn2rank_fac'])


#########################################################################
df=learn2rank_fac
df=convertTimeMulindex(df)
facdata = alphalens.utils.get_clean_factor_and_forward_returns(df['learn2rank_fac'], prices.shift(-1), quantiles=6, periods=(1,3,5),max_loss=1)
lsr=getLongShortRet(facdata,days=1,groups=6,lvg=10,if_vol_premium=False)
plotMeanStd(running_loss['LOSS'],n=100)
plt.show()
plotMeanStd(running_loss['ICIR'],n=100)
plt.show()
plt.hist(score,bins=50)
plt.show()
plotByRet(lsr)
plt.show()


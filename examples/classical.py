import sys
sys.path.append('.')
import joblib
from toolbox import *
import argparse
import alphalens
import os
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



parser = argparse.ArgumentParser()
parser.add_argument('--train_interval',default=600)
parser.add_argument('--repeat_epochs',default=10)
parser.add_argument('--batch_size',default=100)
parser.add_argument('--lr',default=0.05)
parser.add_argument('--train_time_len',default=200)
parser.add_argument('--icir_len',default=20)

args = parser.parse_args()
args.pca_term_fac=joblib.load('output/pca_fac_df.m')

args.main_freq_price_df=joblib.load('output/main_freq_price_df.m')
args.features=joblib.load('output/X.m')
args.ranks=joblib.load('output/y1.m')




df=args.pca_term_fac.groupby('contractObject',as_index=False).rolling(60).mean()
df=convertTimeMulindex(df)

facdata = alphalens.utils.get_clean_factor_and_forward_returns(-df['PCA2'], args.main_freq_price_df.shift(-1), quantiles=5, periods=(1,5,20),max_loss=1)
lsr2=getLongShortRet(facdata,days=1,groups=5,lvg=10,if_vol_premium=False)
plotByRet(lsr2)
plt.show()



df=args.pca_term_fac.groupby('contractObject',as_index=False).rolling(1).mean()
df=convertTimeMulindex(df)

facdata = alphalens.utils.get_clean_factor_and_forward_returns(df['PCA3'], args.main_freq_price_df.shift(-1), quantiles=5, periods=(1,3,5),max_loss=1)
lsr3=getLongShortRet(facdata,days=1,groups=5,lvg=10,if_vol_premium=False)
plotByRet(lsr3)
plt.show()


plotByRet((lsr3+lsr2)/2)
plt.show()
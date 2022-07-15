

#################################### data treat ################################
from copy import deepcopy
from alphalens.performance import mean_return_by_quantile
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io.sql import read_sql_table,to_sql
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base




def getMainFreqPrice(all_market_data,if_tz=True):
    all_market_data_set_date=all_market_data.set_index(['tradeDate']) #set tradeDate as index
    all_market_data_set_date.sort_index(level=['tradeDate'],inplace=True) #order
    #every contractObject and ticker calculate logReturn
    all_market_data_logReturn=all_market_data_set_date.groupby(['contractObject','ticker']).apply(lambda df:np.log(df['closePrice']).diff())
    # transpose to dataframe
    all_market_data_logReturn=pd.DataFrame(all_market_data_logReturn.values,index=all_market_data_logReturn.index,columns=['logReturn'])

    all_market_data_set_mulindex=all_market_data.set_index(['contractObject','ticker','tradeDate'])
    # concat by column
    df=pd.concat([all_market_data_set_mulindex,all_market_data_logReturn],axis=1)
    df=df.loc[df['mainCon'] == 1,['logReturn','closePrice']]
    df=df.reset_index(['ticker','contractObject'])
    df=df.groupby('contractObject')
    mainCon=df.apply(lambda df:np.exp(np.cumsum(df['logReturn']))*df['closePrice'].iloc[0])
    main=mainCon.unstack('contractObject')
    main.index.name='tradeDate'
    if if_tz:
        main.index=map(pd.Timestamp,main.index)
        main.index = main.index.tz_localize('utc')
    return main



def getLongShortRet(factordata,days,groups,lvg=10,if_vol_premium=True):
    rq, std_err = mean_return_by_quantile(factordata, by_date=True)
    lsr=lvg*(rq.loc[groups]-rq.loc[1])/2
    lsrd=lsr[f'{days}D']/days
    if if_vol_premium:
        lsrd+=pow(lsrd,2)/2
    return lsrd


def convertTimeMulindex(fac):
    fac=deepcopy(fac)
    lv0=fac.index.get_level_values('tradeDate')
    lv1=fac.index.get_level_values('contractObject')
    index_t=pd.Index(map(pd.Timestamp,lv0),name='tradeDate')
    index_t=index_t.tz_localize('utc')
    fac.index=[index_t,lv1]
    return fac



def drawdownFunction(series,if_asset=False):
    if if_asset:
        drawdown=series-np.maximum.accumulate(series)
    else:
        cum= series.dropna().cumsum()
        running_max = np.maximum.accumulate(cum)
        drawdown= cum-running_max
    return drawdown

def plotByRet(ret):
    plt.figure()
    ax=ret.cumsum().plot(color='black',lw=1)
    drawdown=drawdownFunction(ret)
    plt.ylabel('cum_log_return')
    ax2 = ax.twinx()
    ax2.fill_between(drawdown.index, drawdown, 0 , drawdown<0, color="LightGrey",alpha=0.3)
    ax2.grid(False)
    plt.ylabel('drawback')





def plotMeanStd(series,n=20):
    series=pd.Series(series)
    mean=series.rolling(n).mean()
    std=series.rolling(n).std()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(series.index,mean.values,color='red')
    ax.fill_between(series.index,mean - 1*std ,mean+ 1*std ,color="red",alpha=0.1,label='$1\sigma$')
    ax.fill_between(series.index,mean - 2*std ,mean + 2*std ,color="orange",alpha=0.1,label='$2\sigma$')
    ax.legend()
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.grid(False)
    return fig,ax



def plotByAsset(asset):
    plt.figure()
    ax=asset.plot(color='black',lw=1)
    drawdown=drawdownFunction(asset,if_asset=True)
    plt.ylabel('asset_total')
    ax2 = ax.twinx()
    ax2.fill_between(drawdown.index, drawdown, 0 , drawdown<0, color="green",alpha=0.4)
    ax2.grid(False)
    plt.ylabel('drawback')




def filterByCode(df,code):
    df=df.loc[df['contractObject'].isin(code)]
    return df





################################# get train data #################################


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt



def get_y(prices,forward):
    ret=np.log(prices).diff().shift(-forward)
    sec_median=pd.Series(np.nanmedian(ret,axis=1),index=ret.index).fillna(0)
    _=[ret.iloc[k].fillna(m,inplace=True) for k,m in enumerate(sec_median)]
    data_y=np.array(ret)
    return data_y



def get_X(pca_term_factor,lens):
    pca_term_factor=pca_term_factor.groupby(level='contractObject',as_index=False).rolling(lens).mean().fillna(0)
    
    if 'contractObject' in pca_term_factor.columns:
        del pca_term_factor['contractObject']
        
    features=pca_term_factor.columns
    dates=np.unique(pca_term_factor.index.get_level_values('tradeDate'))
    codes=np.unique(pca_term_factor.index.get_level_values('contractObject'))
    data_X=np.zeros(shape=(len(dates),len(codes),len(features)))

    for k,d in enumerate(dates):
        df=pca_term_factor.loc[d]
        data_X[k]=df
    return data_X


def get_X_seq(pca_term_factor,lens):
    features=pca_term_factor.columns
    dates=np.unique(pca_term_factor.index.get_level_values('tradeDate'))
    codes=np.unique(pca_term_factor.index.get_level_values('contractObject'))
    data_X=np.zeros(shape=(len(dates),lens,len(codes),len(features)))

    for k,d in enumerate(dates):
        if k>= lens:
            for day_ind,his_day in enumerate(dates[k-lens+1:k+1]):
                df=pca_term_factor.loc[his_day]
                data_X[k,day_ind,:,:]=df
    return data_X.swapaxes(1,2)








############################################ net train ###############################################
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch
from torch import nn, optim
import torch.nn.functional as F




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        m.weight.data.normal_(mean = 0, std = 0.5)
        m.bias.data.fill_(0.05)


class Closs(nn.Module):
    def __init__(self):
        super(Closs, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.sum(f[:,num_stocks // 2:], dim = 1) - torch.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            l += torch.logsumexp(f[:,i:num_stocks-i], dim = 1)
            l += torch.logsumexp(torch.neg(f[:,i:num_stocks-i]), dim = 1)
        l = torch.mean(l)
        return l

class Closs_explained(nn.Module):
    def __init__(self):
        super(Closs_explained, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.sum(f[:,num_stocks // 2:], dim = 1) - torch.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            subtract = torch.tensor(num_stocks - 2*i,requires_grad = False)
            l += torch.log(torch.sum(torch.exp(f[:,i:num_stocks-i]), dim = 1)*torch.sum(torch.exp(torch.neg(f[:,i:num_stocks-i])), dim = 1)-subtract)
        l = torch.mean(l)
        return l

class Closs_sigmoid(nn.Module):
    def __init__(self):
        super(Closs_sigmoid, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.tensor(1, requires_grad=False)+torch.exp(f[:,num_stocks//2:] - f[:,:num_stocks//2])
        return torch.mean(torch.log(l))

class Lloss(nn.Module):
    def __init__(self):
        super(Lloss, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.neg(torch.sum(f, dim = 1))
        for i in range(num_stocks):
            l += torch.logsumexp(f[:,i:], dim = 1)
        l = torch.mean(l)
        return l






class ICloss(nn.Module):
    def __init__(self):
        super(ICloss, self).__init__()
    def forward(self, f, num_stocks):
        sort=torch.arange(num_stocks,0,-1,requires_grad=False).unsqueeze(0).repeat(f.shape[0],1).double()
        f_mean=f.mean(axis=1).unsqueeze(1).repeat(1,num_stocks)
        sort_mean=sort.mean(axis=1).unsqueeze(1).repeat(1,num_stocks)
        f_std=f.std(axis=1)
        sort_std=sort.std(axis=1)
        ic=torch.neg(torch.mean((f-f_mean)*(sort-sort_mean), dim = 1))/(f_std*sort_std+0.001)
        ic_mean=ic.mean()
        return ic_mean


class ICIRloss(nn.Module):
    def __init__(self):
        super(ICIRloss, self).__init__()
    def forward(self, f, num_stocks):
        sort=torch.arange(num_stocks,0,-1).unsqueeze(0).repeat(f.shape[0],1).double()
        f_mean=f.mean(axis=1).unsqueeze(1).repeat(1,num_stocks)
        sort_mean=sort.mean(axis=1).unsqueeze(1).repeat(1,num_stocks)
        f_std=f.std(axis=1)
        sort_std=sort.std(axis=1)
        ic=torch.neg(torch.mean((f-f_mean)*(sort-sort_mean), dim = 1))/(f_std*sort_std+0.001)
        ic_mean=ic.mean()/ic.std()
        return ic_mean




class CMLE(nn.Module):
    def __init__(self, n_features):
        super(CMLE, self).__init__()
        self.n_features = n_features
        self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
        self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
        self.linear3 = nn.Linear(self.n_features * 2, self.n_features // 2)
        self.linear4 = nn.Linear(self.n_features // 2, 1)
        self.apply(weights_init)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        result = F.relu(self.linear4(x))
        result = result.view(result.shape[0], result.shape[1])
        return result




class LMLE(nn.Module):
    def __init__(self, n_features, num_stocks):
        super(LMLE, self).__init__()
        self.n_features = n_features
        self.num_stocks = num_stocks
        self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
        self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
        self.linear3 = nn.Linear(self.n_features * 2, self.n_features // 2)
        self.linear4 = nn.Linear(self.n_features // 2, 1)
        self.apply(weights_init)
    
    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        result = F.sigmoid(self.linear4(x))
        result = result.view(result.shape[0], result.shape[1])
        return result





class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.state_dim, self.action_dim=input_size, output_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout=dropout,
            batch_first=True
        )
        self.linear_in = nn.Sequential(nn.Linear(hidden_size, 8), nn.LeakyReLU())


        self.net = nn.Sequential(nn.Linear(8, 5), nn.LeakyReLU(),
                            nn.Linear(5, 5),nn.LeakyReLU(),
                            nn.Linear(5, 8))

        self.linear_out = nn.Sequential(nn.LeakyReLU(),nn.Linear(8, output_size),nn.Softmax(dim=1))

    def forward(self, x):
        size=x.shape
        x=x.reshape(size[0]*size[1],size[2],size[3])
        output, (h_n, c_n) = self.lstm(x)
        output=output.reshape(size[0],size[1],size[2],self.hidden_size)
        o=self.linear_in(output[:,:, -1, :])
        x = self.net(o)
        pred = self.linear_out(x).squeeze(2)
        return pred

    def init_hidden(self):
        return torch.randn(1, 12, self.hidden_size)





def return_rank(a):
    a = a * -1
    order = a.argsort()
    return order.argsort()

def random_batch(x, y, batch_size):
    
	ind = np.random.randint(0, len(x),batch_size)
	batch_x, batch_y = x[ind], y[ind]
	x_sorted = np.zeros(batch_x.shape)
	for i in range(len(batch_x)):
		rank_temp = return_rank(batch_y[i])
		rank2ind = np.zeros(x.shape[1], dtype = int)
		for j in range(len(rank_temp)):
			rank2ind[rank_temp[j]] = int(j)
		for j in range(len(rank_temp)):
			x_sorted[i,rank_temp[j],:] = batch_x[i][rank2ind[rank_temp[j]]]
	return x_sorted








def train(features, ranks, model,loss,batch_size,epochs,lr=0.01):
    print('Done reading data\n')
    opt = optim.Adam(model.parameters(), lr=lr)
    print('Done building model\n')
    ICIRloss_fun=ICIRloss()
    running_loss = []
    torch.set_grad_enabled(True)
    for itr in range(epochs):
        batch_x = Variable(torch.from_numpy(random_batch(features, ranks,batch_size)).double())
        model.train()
        scores = model(batch_x)
        l = loss(scores, torch.tensor(batch_x.shape[1], requires_grad = False))
        ICIR_data = -ICIRloss_fun(scores, torch.tensor(batch_x.shape[1], requires_grad = False))

        opt.zero_grad()

        l.backward()
        opt.step()

        running_loss.append([float(l),float(ICIR_data)])

        if (itr+1) % 10 == 0:
            loss_,icir_=np.mean(running_loss,axis=0)
            print("step", (itr+1), f'loss: {loss_:.4f}, ICIR: {icir_:.4f}')
            
    running_loss=pd.DataFrame(running_loss,columns=['LOSS','ICIR'])    
    return model,running_loss




def prior_data_random_batch(x, y,prior_data,batch_size):
    ind = np.random.randint(0, len(x),batch_size)
    batch_x, batch_y, batch_prior_data= x[ind], y[ind], prior_data[ind]
    x_sorted = np.zeros(batch_x.shape)
    prior_data_sorted=np.zeros(batch_prior_data.shape)

    for i in range(len(batch_x)):
        rank_temp = return_rank(batch_y[i])
        rank2ind = np.zeros(x.shape[1], dtype = int)
        for j in range(len(rank_temp)):
            rank2ind[rank_temp[j]] = int(j)
        for j in range(len(rank_temp)):
            x_sorted[i,rank_temp[j],:] = batch_x[i][rank2ind[rank_temp[j]]]
            prior_data_sorted[i,rank_temp[j]]=batch_prior_data[i][rank2ind[rank_temp[j]]]
    return x_sorted,prior_data_sorted






def prior_data_train(features, ranks,prior_data, model,loss,batch_size,epochs,lr=0.01):
    print('Done reading data\n')
    opt = optim.Adam(model.parameters(), lr=lr)
    print('Done building model\n')
    ICIRloss_fun=ICIRloss()
    running_loss = []
    torch.set_grad_enabled(True)
    for itr in range(epochs):
        batch_x_,prior_data_sorted_=prior_data_random_batch(features, ranks,prior_data,batch_size)
        batch_x = Variable(torch.from_numpy(batch_x_).double())
        prior_data_sorted=torch.tensor(prior_data_sorted_, requires_grad = False)
        model.train()
        scores = model(batch_x)+prior_data_sorted
        l = loss(scores, torch.tensor(batch_x.shape[1], requires_grad = False))
        ICIR_data = -ICIRloss_fun(scores, torch.tensor(batch_x.shape[1], requires_grad = False))

        opt.zero_grad()

        l.backward()
        opt.step()

        running_loss.append([float(l),float(ICIR_data)])

        if (itr+1) % 10 == 0:
            loss_,icir_=np.mean(running_loss,axis=0)
            print("step", (itr+1), f'loss: {loss_:.4f}, ICIR: {icir_:.4f}')
            
    running_loss=pd.DataFrame(running_loss,columns=['LOSS','ICIR'])    
    return model,running_loss
















def nonrandom_batch(x, y):
	batch_x, batch_y = x, y
	x_sorted = np.zeros(batch_x.shape)
	for i in range(len(batch_x)):
		rank_temp = return_rank(batch_y[i])
		rank2ind = np.zeros(x.shape[1], dtype = int)
		for j in range(len(rank_temp)):
			rank2ind[rank_temp[j]] = int(j)
		for j in range(len(rank_temp)):
			x_sorted[i,rank_temp[j],:] = batch_x[i][rank2ind[rank_temp[j]]]
	return x_sorted



def metric(features, ranks, model,loss):
    ICIRloss_fun=ICIRloss()
    torch.set_grad_enabled(False)
    batch_x = Variable(torch.from_numpy(nonrandom_batch(features, ranks)).double())
    scores = model(batch_x)
    l = loss(scores, torch.tensor(batch_x.shape[1], requires_grad = False))
    ICIR_data = -ICIRloss_fun(scores, torch.tensor(batch_x.shape[1], requires_grad = False))
    return float(l),float(ICIR_data)





################################# bt #############################################
from backtrader.feeds import PandasData
import backtrader as bt
from datetime import datetime
from datetime import datetime

def get_quote_dict(all_market_data):
    dic={}
    df=all_market_data
    old_col=['tradeDate','openPrice','highestPrice','lowestPrice','closePrice','turnoverVol','openInt','contractObject','mainCon','smainCon','ticker']
    new_col=['datetime','open','high','low','close','volume','openinterest','product','main','smain','ticker']
    df=df.loc[:,old_col]
    df.sort_values(by=['ticker','tradeDate'],inplace=True)
    df.columns=new_col
    df['datetime']=df.datetime.astype('datetime64[ns]')
    df=df.dropna()
    for ticker in df.ticker.unique():
        t=df.query(f"ticker == '{ticker}'")
        t=t.reset_index()
        del t['index']
        dic[ticker]=t
    return dic



class query_main:
    def __init__(self,all_market_data):
        df=all_market_data.copy()
        df=df.loc[(df['mainCon'] == 1) | (df['smainCon'] == 1),['tradeDate','contractObject','ticker','mainCon','smainCon']]
        df['tradeDate']=df.tradeDate.astype('datetime64[ns]')
        self.all_market_data=df
    def get_main(self,date):
        df=self.all_market_data
        df=df.loc[(df['tradeDate'] == date) & (df['mainCon'] == 1)]
        return df.ticker.values.tolist()
    def get_sub_main(self,date):
        df=self.all_market_data
        df=df.loc[(df['tradeDate'] == date) & (df['smainCon'] == 1)]
        return df.ticker.values.tolist()



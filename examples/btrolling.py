import sys
sys.path.append('.')
import joblib
from toolbox import *
import argparse
import os
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--train_interval',default=1000000000)
parser.add_argument('--repeat_epochs',default=300)
parser.add_argument('--batch_size',default=100)
parser.add_argument('--lr',default=0.01)
parser.add_argument('--train_time_len',default=120)
parser.add_argument('--icir_len',default=20)

args = parser.parse_args()
args.net=[]
args.data=joblib.load('output/all_market_data.m')
args.features=joblib.load('output/X.m')
args.ranks=joblib.load('output/y1.m')

args.model=SimpleLSTM(input_size=2,hidden_size=5).double()
args.loss=Lloss()
args.query_main=query_main(args.data)



df=joblib.load('output/pca_fac_including_all_code_df.m')[['PCA2']]
args.prior_fac=100000*nn.Softmax(dim=1)(torch.tensor(-get_X(df,60)[:,:,0], requires_grad = False))
args.prior_fac=torch.tensor(-get_X(df,60)[:,:,0], requires_grad = False)







class SimpleStrategy(bt.Strategy):
    def __init__(self):
        args.net=[]
        self.model=None
        self.predict_score=np.zeros(args.ranks.shape)
        self.date_ind=0


    def next(self):     
        now = self.lines.datetime.datetime(0)
        main_code_list = args.query_main.get_main(date=now)

###################### cal ###################################

        time_len=min(self.date_ind,args.train_time_len-1)+1
        train_X=args.features[self.date_ind-time_len+1:self.date_ind+1]
        prior_data=args.prior_fac[self.date_ind-time_len+1:self.date_ind+1]
        train_y=args.ranks[self.date_ind-time_len+1:self.date_ind+1]

        metric_X=args.features[self.date_ind-args.icir_len + 1:self.date_ind]
        metric_y=args.ranks[self.date_ind-args.icir_len + 1:self.date_ind]

    ########### predict by icir weighted ###########
        pred=args.prior_fac[self.date_ind].numpy()#np.zeros((args.ranks.shape[1],))
        if self.model is not None:
            _,icir=metric(metric_X, metric_y, self.model ,args.loss)
            print(f'模型: icir={icir}')
            lstm_pred=self.model(args.features[self.date_ind:self.date_ind+1]).detach().numpy().flatten()
            pred += lstm_pred
            

        ############ train  ... ###################
        if (self.date_ind+1) % args.train_interval == 0 and time_len >= args.batch_size:
            print(f'now is {self.date_ind+1},{now}')
            self.model,_=prior_data_train(train_X,train_y,prior_data,deepcopy(args.model),args.loss,args.batch_size,args.repeat_epochs,args.lr)
            

        ########################## trade ... ######################
        if len(set(pred)) > args.ranks.shape[1]//2:
            factor_scores = pred.argsort().argsort()/(len(pred)-1)
            if self.date_ind % 1 == 0:
                print(main_code_list,pred)
            



            for main_code,factor_score in zip(main_code_list,factor_scores):

                main_code_data = self.getdatabyname(main_code)		


                if factor_score >= 0.8:	
                    #self.order_target_size(data=main_code_data,target=1)
                    self.order_target_percent(data=main_code_data,target=0.01)


                elif factor_score <= 0.2:
                    #self.order_target_size(data=main_code_data,target=-1)
                    self.order_target_percent(data=main_code_data,target=-0.01)

                else:
                    #self.order_target_size(data=main_code_data,target=0)
                    self.order_target_percent(data=main_code_data,target=0)
        else:
            for pos_data in self.broker.positions:
                #self.order_target_size(data=pos_data,target=0)
                self.order_target_percent(data=pos_data,target=0)            

      
        for pos_data in self.broker.positions:
            if pos_data._name not in main_code_list:
                #self.order_target_size(data=pos_data,target=0)
                self.order_target_percent(data=pos_data,target=0)
                self.order_target_value(data=pos_data,target=0)##################################################################################


########################### update #########################
        args.net.append(self.broker.getvalue())	
        print(f'{now}: 当前资金:{self.broker.getvalue()}')	
        self.date_ind+=1	

    def prenext(self):
        self.next()

    def stop(self):
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')




cerebro = bt.Cerebro()
data_dict=get_quote_dict(args.data)
for key,value in data_dict.items():
    data = PandasData(dataname=value, datetime='datetime')
    cerebro.adddata(data,name=key)





cerebro.broker.setcommission(commission=0.0000, margin=200000, mult=10)


cerebro.broker.set_coc(True)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name = 'SharpeRatio')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DW')
cerebro.addstrategy(SimpleStrategy)
cerebro.broker.setcash(100000000)
r=cerebro.run()




strat = r[0]
print('最终资金: %.2f' % cerebro.broker.getvalue())
print('夏普比率:', strat.analyzers.SharpeRatio.get_analysis())
print('回撤指标:', strat.analyzers.DW.get_analysis())



ind=np.unique(args.data.tradeDate.astype('datetime64[ns]'))
plotByAsset(pd.Series(args.net,index=ind))
plt.show()







b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
cerebro.plot(b)
plt.show()


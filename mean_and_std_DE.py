import numpy as np
import matplotlib.pyplot as plt
import Coded_Algorithms.Algorithms as alg
import Coded_Algorithms.hybrid_algorithms as hyb
import Functions as fun
import pandas as pd

bounds = np.asarray([[0,3.14]]*10)
dimension = len(bounds)
de_evals = 100000
algt = "HDE"
runs = 5
x = 498
de_timings, de_objs = np.ndarray([runs,x]), np.ndarray([runs,x])
class save_csv():
    def __init__(self,csv_id):
        self.data = list()
        self.csv_id = csv_id
    def prepare_data(self,std,mean,f_eval,curr_alg):
            self.data = [std,mean,f_eval]

    def store_data(self):
        df_rates = pd.DataFrame.from_dict(self.data)
        df_rates.to_csv(self.csv_id,index=False) 

csv_id = algt + "_new.csv"
save_data = save_csv(csv_id)


for run in range(runs):
    [obj_class, obj_track, obj_eval, timing] = hyb.hybrid_genetic(fun.Michalewitz, bounds, 100000, de_evals,tol=0)
    print(obj_track[-1])
    de_objs[run,:] = obj_track
    de_timings[run,:] = timing
    print("running")


de_std = np.ndarray([len(de_objs[0,:])])
de_timing = np.ndarray([len(de_objs[0,:])])
de_mean = np.ndarray([len(de_objs[0,:])])

de_mean = np.apply_along_axis(np.mean,0,de_objs)
de_timing = np.apply_along_axis(np.mean,0,de_timings)
de_std = np.apply_along_axis(np.std,0,de_objs)

save_data.prepare_data(de_std,de_mean,obj_eval,algt)
save_data.store_data() 

""" plt.plot(de_timing,de_mean)
#plt.plot(sa_mean,sa_timing)
plt.yscale('log')
plt.xlabel("Time")
#plt.xlim((0,10))
plt.fill_between(de_timing, de_mean-de_std, de_mean+de_std, alpha=0.2)
#plt.fill_between(sa_mean, -sa_std, sa_std, alpha=0.2)
plt.ylim((0.8, 10e2))
plt.ylabel("Objective Function")
plt.legend(["DE","SA"])
plt.title(f' Objective Function vs Time for {dimension} Dimensions')
plt.show() """

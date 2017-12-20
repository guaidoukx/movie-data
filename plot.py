import matplotlib.pyplot as plt

yy_pred = gbdty_pred
yy_real = gbdty_test
def sample(y):
    import random
    index=random.sample(range(len(y)),800)  
    return index

index = sample(yy_pred)

y_pred_sample = yy_pred[index]
y_real_sample = yy_real.iloc[index]

samples = zip(y_real_sample,y_pred_sample)
samples = sorted(samples)
# samples = sorted(samples,key=lambda samples : samples[1])

y_real_sample_sorted = []
y_pred_sample_sorted = []
for sample in samples:
    y_real_sample_sorted.append(sample[0])
    y_pred_sample_sorted.append(sample[1])
# print(y_real_sample_sorted)
# print(y_pred_sample_sorted)

plt.figure()
plt.plot(range(len(y_real_sample_sorted)),y_pred_sample_sorted,'b',label="predict")
plt.plot(range(len(y_real_sample_sorted)),y_real_sample_sorted,'r',label="real")
plt.legend(loc="upper right") # 显示图中的标签
plt.xlabel("movies")
plt.ylabel('vote average')
plt.show()
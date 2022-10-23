import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad

def kde(times, X, change_start_time):
    """
    作用：用KDE方法计算某条时间序列change_start_time后overflow和underflow的概率
    参数：
    times: timestamp序列，array
    X: times对应的值序列，array
    change_start_time: 变化开始时间，timestamp，注意这里的change_start_time是绝对时间
    返回：
    po: overflow概率
    pu: underflow概率
    """
    # 找到times里面对应change_start_time的下标
    pos = np.argwhere(times == change_start_time)[0][0]

    # bandwidth
    width = 0.1
    # X标准化，[0,1]之间
    X_min = np.min(X)
    X_max = np.max(X)
    X_std = (X - X_min) / (X_max - X_min)
    X_train = X_std[:pos]
    X_test = X_std[pos:]
    train_value = X_train.reshape(-1, 1)

    # 建立KDE模型
    d = KernelDensity(kernel='gaussian', bandwidth=width).fit(train_value)

    # 计算概率值
    po = 0
    pu = 0
    for t in range(len(X_test)):
        # overflow 概率
        prob_o = quad(lambda x: np.exp(d.score_samples(np.array(x).reshape(-1, 1))), X_test[t], 5)[0]
        # underflow 概率
        prob_u = quad(lambda x: np.exp(d.score_samples(np.array(x).reshape(-1, 1))), -5, X_test[t])[0]
        po += np.log(prob_o)
        pu += np.log(prob_u)
    po = -po / len(X_test)
    pu = -pu / len(X_test)

    return po, pu


def change_start_timestamp(times, values, etimestamp):
    """
    作用：
    针对给定的一段时间序列和错误时间点，一阶差分后用3-sigma法（这里取2.5-sigma）找到异常开始时间，
    对于用3-sigma无法确定异常发生时间的时间序列（异常不明显或根本没有），
    找其偏离均值最大的点作为异常开始时间，顺便返回KDE之后的overflow和underflow值
    参数：
    times: 时间序列的timestamp数列，np.array
    values: 时间序列的value数列，np.array
    etimestamp: 时间序列的error_timestamp，int
    返回：
    一个list：3个元素
    [change_start: 异常开始时间戳-etimestamp，int
    po: overflow 概率, float
    pu: underflow 概率, float]
    """
    if len(values)>1:
        # 一阶差分
        times = times[1:]
        div_values = values[1:] - values[:values.shape[0] - 1]
    
        # 3-sigma法则
        std_values = np.std(div_values)
        mean_values = np.mean(div_values)
        up_thres = mean_values + 2.5 * std_values
        adjusted_div_values = abs(div_values - mean_values)
        bool_array = adjusted_div_values > up_thres
        # plt.plot(times,div_values)
        # 如果没有可能异常点，返回0
        if sum(bool_array) == 0 or std_values == 0:
            # 返回第一个最大值所在的位置下标
            pos = np.argmax(adjusted_div_values)
            change_start_time = times[pos]
        else:
            # 从可疑点中找距离error_timestamp最近的时间点(此处可能会有问题，如果最近的是change end time怎么办？？？)
            minindex = np.argmin(abs(times[bool_array] - etimestamp))
            change_start_time = times[bool_array][minindex]
        
        if len(times[times < change_start_time]) >= 2 and len(times[times > change_start_time]) >= 2:
            po, pu = kde(times, div_values, change_start_time)
            return [change_start_time-etimestamp, po, pu]

    return [0, 0, 0]



if __name__=='__main__':
    # 测试kde函数
    times = np.array([1,2,3,4,5,6,7])
    X = np.array([1,2,1,20,1,2,1])
    change_start_time = 4
    kde(times, X,change_start_time)
    
    
    # 测试change_start_time 函数
    times = np.array([1,2,3,4,5,6,7])
    X = np.array([1,2,1,20,1,2,1])
    etimestamp = 4
    change_start_timestamp(times, X, etimestamp)

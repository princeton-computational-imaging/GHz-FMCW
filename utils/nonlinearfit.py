
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import least_squares

################### nonlinear fitting functions ###################
################### nonlinear fitting functions ###################
################### nonlinear fitting functions ###################

def raw_model(x, t):
    return x[0]*np.sin(2*np.pi*t*x[1]+x[2])+x[3] #*pow(t, 3) + x[7]*pow(t, 2) + x[8]*t + x[9]

def complete_model(x, t):
    return (x[0]*pow(t, 2) + x[1]*t + x[2])*np.sin(2*np.pi*t*x[3]+x[4]) + x[5]*pow(t, 3) + x[6]*pow(t, 2) + x[7]*pow(t, 1) + x[8]

def raw_fun(x, t, y):
    return raw_model(x, t) - y

def complete_fun(x, t, y):
    return complete_model(x, t) - y

def nonlin_fit_complete(data_stack, delta_d, plot_flag):
    fit_params = []
    
    p0 = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    p8 = []
    
    
    dist = 400
    t = np.linspace(0, 0.0000325, len(data_stack[0]), endpoint=True)

    for data in data_stack:
        x0_raw = [0.04,137000,3.14,0]
        y = data
        raw_res = least_squares(raw_fun, x0_raw, method = 'lm', ftol=2.23e-16, xtol=2.23e-16, gtol=2.23e-16, \
                                max_nfev=5000, args=(t, y), verbose=plot_flag)
        y_test_raw = raw_model(raw_res.x, t)

        x0_complete = [0,0,raw_res.x[0],raw_res.x[1],raw_res.x[2]%(2*np.pi),raw_res.x[3], 0, 0, 0]
        complete_res = least_squares(complete_fun, x0_complete, method = 'lm', ftol=2.23e-16, xtol=2.23e-16, gtol=2.23e-16, \
                                     max_nfev=5000, args=(t, y), verbose=plot_flag)
        y_test_complete = complete_model(complete_res.x, t)

        p0.append(complete_res.x[0])
        p1.append(complete_res.x[1])
        p2.append(complete_res.x[2])
        p3.append(complete_res.x[3])
        p4.append(complete_res.x[4])
        p5.append(complete_res.x[5])
        p6.append(complete_res.x[6])
        p7.append(complete_res.x[7])
        p8.append(complete_res.x[8])
        
        
        if plot_flag == 1:
            print("raw fit params: ")
            print(raw_res.x)
            print("complete fit params: ")
            print(complete_res.x)

            plt.plot(t, y, 'r', markersize=0.5, label='data')
            plt.plot(t, y_test_complete, 'g', label='fitted complete model')
            plt.xlabel("t")
            plt.ylabel("y")
            plt.title("r1 "+str(dist))
            plt.legend(loc='lower right')
            plt.show()
        
        print(str(dist)+'/'+str(len(data_stack)+400))
        dist = dist + delta_d
    
    fit_params.append(p0)
    fit_params.append(p1)
    fit_params.append(p2)
    fit_params.append(p3)
    fit_params.append(p4)
    fit_params.append(p5)
    fit_params.append(p6)
    fit_params.append(p7)
    fit_params.append(p8)
        
    return np.array(fit_params)

def nonlin_fit(data_stack, delta_d, plot_flag):
    fit_params = []
    
    r1_p2 = []
    r1_p1 = []
    r1_p0 = []
    freq = []
    phase = []
    r1_c5 = []
    r1_c6 = []
    r1_c7 = []
    r1_c8 = []
    
    dist = 400
    t = np.linspace(0, 0.0000325, len(data_stack[0]), endpoint=True)

    for data in data_stack:
        x0_raw = [0.04,137000,3.14,0]
        y = data
        raw_res = least_squares(raw_fun, x0_raw, method = 'lm', ftol=2.23e-16, xtol=2.23e-16, gtol=2.23e-16, max_nfev=5000, args=(t, y), verbose=plot_flag)
        y_test_raw = raw_model(raw_res.x, t)

        x0_complete = [0,0,raw_res.x[0],raw_res.x[1],raw_res.x[2]%(2*np.pi),raw_res.x[3], 0, 0, 0]
        complete_res = least_squares(complete_fun, x0_complete, method = 'lm', ftol=2.23e-16, xtol=2.23e-16, gtol=2.23e-16, max_nfev=5000, args=(t, y), verbose=plot_flag)
        y_test_complete = complete_model(complete_res.x, t)

        r1_p2.append(complete_res.x[0])
        r1_p1.append(complete_res.x[1])
        r1_p0.append(complete_res.x[2])
        freq.append(complete_res.x[3])
        phase.append(complete_res.x[4])
        r1_c5.append(complete_res.x[5])
        r1_c6.append(complete_res.x[6])
        r1_c7.append(complete_res.x[7])
        r1_c8.append(complete_res.x[8])
        
        
        if plot_flag == 1:
            print("raw fit params: ")
            print(raw_res.x)
            print("complete fit params: ")
            print(complete_res.x)

            plt.plot(t, y, 'r', markersize=0.5, label='data')
            #plt.plot(t, y_test_raw, 'b', label='fitted raw model')
            plt.plot(t, y_test_complete, 'g', label='fitted complete model')
            plt.xlabel("t")
            plt.ylabel("y")
            plt.title("r1 "+str(dist))
            plt.legend(loc='lower right')
            plt.show()
        
        print(str(dist)+'/'+str(len(data_stack)+400))
        dist = dist + delta_d
    
    
    fit_params.append(freq)
    fit_params.append(phase)
    
    
    
    if plot_flag == 1:
        param_x = np.linspace(1,len(fit_params[0]),len(fit_params[0]))
        
        plt.plot(param_x, fit_params[3], label='fitted model')
        plt.xlabel("x")
        plt.ylabel("freq")
        plt.show()
        
        plt.plot(param_x, fit_params[4], label='fitted model')
        plt.xlabel("x")
        plt.ylabel("phase")
        plt.show()
        
    return np.array(fit_params)
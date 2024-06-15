import numpy as np
import math

def Cauchy(x,fwhm,height,center,offset):
    f =  (height)*(fwhm/2)**2/((fwhm/2)**2+(x-center)**2)+offset
    return f

def CauchyNoOffset(x,fwhm,height,center):
    f =  (height)*(fwhm/2)**2/((fwhm/2)**2+(x-center)**2)
    return f

def IntofCauchyNoOffset(x,fwhm,height,center):
    f =  (height)*(fwhm/2)*(np.arctan(2*(x-center)/fwhm)+np.pi/2)
    return f


def nanadd(a1,a2):
    summ = np.copy(a1)
    if len(a1)!=len(a2):
        print("Error:array of non-equal length for function nanadd!")
        return
    for ii in range(len(a1)):
        if np.isnan(a1[ii]) and np.isnan(a2[ii]):
            summ[ii] = np.nan
        elif np.isnan(a1[ii]) and ~np.isnan(a2[ii]):
            summ[ii] = a2[ii]
        elif ~np.isnan(a1[ii]) and np.isnan(a2[ii]):
            summ[ii] = a1[ii]
        else:
            print("Error: both elements are non-nans")
            return
    return summ


def get_point(shot_num,num_points):
    return shot_num % num_points


def get_loop(shot_num,num_points):
    return shot_num//num_points


def CrossingOfTwoGaussians(delta1,sigma1,delta2,sigma2):
    if delta1>delta2:
        deltatemp=delta1
        delta1=delta2
        delta2=deltatemp
        sigmatemp=sigma1
        sigma1=sigma2
        sigma2=sigmatemp
    return 1/(sigma1**2 - sigma2**2) *\
(delta2*sigma1**2 - delta1*sigma2**2 - \
   np.sqrt(delta1**2 * sigma1**2 * sigma2**2 - \
    2 * delta1 * delta2 * sigma1**2 * sigma2**2 +\
           delta2**2 * sigma1**2 * sigma2**2 - \
    2 * sigma1**4 * sigma2**2 * np.log(sigma2/sigma1) + \
    2 * sigma1**2 * sigma2**4 * np.log(sigma2/sigma1)))


def overlapoftwoGaussians(delta1,sigma1,delta2,sigma2):
    crossing = CrossingOfTwoGaussians(delta1,sigma1,delta2,sigma2)
    leftintegral = (1-math.erf(abs(delta1-crossing)/np.sqrt(2)/sigma1)) / 2
    rightintegral = (1-math.erf(abs(delta2-crossing)/np.sqrt(2)/sigma2)) / 2
    return (leftintegral+rightintegral)


def zero_crossing(x_array, y_array):
    sign_array = np.sign(y_array)
    idx = np.nonzero(sign_array[1:]-sign_array[:-1])[0][0]
    zero_point = (idx*y_array[idx+1]-(idx+1)*y_array[idx])/\
    (y_array[idx+1]-y_array[idx])
    return zero_point*(x_array[idx+1]-x_array[idx]) + (idx+1)*x_array[idx] - idx*x_array[idx+1]
import numpy as np
from scipy import optimize
def gaussian2D(height, center_x, center_y, width_x):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    # width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_x)**2)/2)

def gaussian1D(height, center, width):
    """Returns a gaussian function with the given parameters"""
    width = float(width)
    
    return lambda x: height*np.exp(
                -(((center-x)/width)**2)/2)

def lorentzian1D(height, center, width, y0, slope):
    """Returns a gaussian function with the given parameters"""
    width = float(width)
    
    return lambda x: height*( width/((center-x)**2+(width)**2) ) + y0 + x*slope


def lorentzian2D(height, center_x, center_y, width_x, a, b, c):
    """Returns a lorenztian function with the given parameters"""
    width_x = float(width_x)

    return lambda y,x: height*( width_x/((center_x-x)**2+(center_y-y)**2+(width_x)**2)) + a*x+b*y+c

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    Y, X = np.indices(data.shape)
    y = (Y*data).sum()/total
    x = (X*data).sum()/total
    col = data[:, int(x)]
    width_y = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(y), :]
    width_x = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return (height, x, width_x), (height, y, width_y)

def fitPeak2D(data, func):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(func(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, (params[0][0], params[0][1], params[1][1], params[0][2], 0, 0, 0))
    return p


def fitPeak(data, func):
    px, py = moments(data)
    xproj = np.sum(data, axis=0)
    yproj = np.sum(data, axis=1)
    xerrorfunction = lambda p: func(*p)(np.arange(xproj.size)) - xproj
    yerrorfunction = lambda p: func(*p)(np.arange(yproj.size)) - yproj
    (xh,xc,xw, y0, slope), success = optimize.leastsq(xerrorfunction, (*px, 0,0))
    (yh,yc,yw, y0, slope), success = optimize.leastsq(yerrorfunction, (*py, 0,0))
    return np.array([xh,xc,xw, yh,yc,yw])
    
def gaussian2D(height, center_y, center_x, width_x):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    # width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_x)**2)/2)
def fitPeak2DGauss(data, func):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(func(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, (params[0][0], params[0][1], params[1][1], params[0][2]))
    return p

def main():
    pass


if __name__ == "__main__":
    main()
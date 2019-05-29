import datetime
import functools

import pandas as pd
import numba
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.optimize import fmin

def convert_down_density(rho_imperial):
    return 1/(rho_imperial)

def convert_fabric_density(lambda_imperial):
    return lambda_imperial/36**2

class Baffle:
    
    def __init__(self, width, height, 
                 top_fabric_weight = 1.1, 
                 bottom_fabric_weight = 1.1, 
                 baffle_fabric_weight = 0.5, 
                 down_fill_power = 800):
        self.fabric_width: float = width
        self.fabric_height: float = height
        self._top_fabric: float = convert_fabric_density(top_fabric_weight)
        self._bottom_fabric: float = convert_fabric_density(bottom_fabric_weight)
        self._baffle_fabric: float = convert_fabric_density(baffle_fabric_weight)
        self._down_density: float = convert_down_density(down_fill_power)
        self.radius = self._opt_radius()
    
    @property
    def area(self):
        return self._baffle_area(self.radius)[0]
    
    @property
    def width(self):
        return self._baffle_width(self.radius)[0]
    
    @property
    def mass(self):
        return (self._down_density*self.area +
               (self._top_fabric + self._bottom_fabric) * self.fabric_width +
               self._baffle_fabric*self.fabric_height)
    
    @property
    def thermal_res(self):
        return (1/(quad(self._inv_baffle_height,
                         -self.width/2,
                         self.width/2)[0]))
    
    @property
    def abs_efficiency(self):
        thermal_res_width_normed = (self.width * self.thermal_res)
        mass_width_normed = (1 / self.width * self.mass)
        return (thermal_res_width_normed / mass_width_normed)
    
    def polygon(self, x=None):
        pass
    
    def mask_above(self, x, y):
        pass
    
    def mask_below(self, x, y):
        pass
    
    def quilt(self, n):
        poly = self.polygon()
        xy = poly.get_xy()
        centers = np.arange(-self._shift*(n-1)/2, self._shift*(n+1)/2, self._shift)
        polys = []
        for i,center in enumerate(centers):
            vertices = (xy + np.array([center, 0])) * np.array([1, np.power(-1, i)])
            polys.append(plt.Polygon(vertices, facecolor=None, fill=False))
        return polys
    
    def _baffle_width(self, radius):
        return 2*radius*np.sin(self.fabric_width/(2*radius))
    
    def _opt_radius(self):
        return minimize(lambda radius: -1*self._baffle_area(radius), 2).x
    
    def _arc_area(self, radius):
        return .5*radius**2*( self.fabric_width/radius - np.sin(self.fabric_width/radius) )
    
class RectBaffle(Baffle):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shift = self.width
    
    def _baffle_area(self, radius):
        return 2*self._arc_area(radius) + self.fabric_height*self._baffle_width(radius)
    
    def _baffle_height(self, x):
        return (np.sqrt(self.radius**2 - x**2) - self.radius*np.cos(self.fabric_width/(2*self.radius)))
    
    def _inv_baffle_height(self, x):
        return 1/(2*self._baffle_height(x) + self.fabric_height)
    
    def polygon(self, x=None):
        if not x:
            x = np.linspace(-self.width/2, self.width/2, 101)
        top = self._baffle_height(x) + self.fabric_height/2
        top = top.reshape((len(x), 1))
        x = x.reshape(len(x), 1)
        bottom = -top
        tops = np.concatenate((x, top), axis=1)
        bottoms = np.concatenate((np.flip(x, 0), bottom), axis=1)
        vertices = np.concatenate((tops, bottoms), axis=0)
        return plt.Polygon(vertices, facecolor=None, fill=False)
    
    def above_mask(self, x, y):
        top = self._baffle_height(x) + self.fabric_height/2
        return (y > top).T

    def below_mask(self, x, y):
        top = self._baffle_height(x) + self.fabric_height/2
        return (y < -top).T
    
    
class TriangleBaffle(Baffle):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = np.array([self.fabric_height])
        self._shift = self.width/2
        
    @property
    def mass(self):
        return (2*self._down_density*self.area +
               (self._top_fabric + self._bottom_fabric) * self.fabric_width +
               2*self._baffle_fabric*self.fabric_height)
    
    def _baffle_area(self, radius):
        return radius*self.fabric_width/2
    
    def _triangle_height(self, radius):
        return np.sqrt(self.fabric_height**2 - (self._baffle_width(radius)/2)**2)
    
    def _arc_height(self, x):
            return np.sqrt(self.radius**2 - x**2) - self.radius*np.cos(self.fabric_width/(2*self.radius))
    
    def _baffle_height(self, x):
        return (self._triangle_height(self.radius) + 
                self._arc_height(x) + self._arc_height(np.abs(x)-self.width/2))
    
    def _inv_baffle_height(self, x):
        return 1/(self._baffle_height(x))
    
    def polygon(self, x=None):
        if not x:
            x = np.linspace(-self.width/2, self.width/2, 101)
        top = self._arc_height(x) + self._triangle_height(self.radius)/2
        top = top.reshape((len(x), 1))
        x = x.reshape(len(x), 1)
        bottom = np.array([[0, -self._triangle_height(self.radius)[0]/2]])
        # print(x,top,bottom)
        tops = np.concatenate((x, top), axis=1)
        bottoms = bottom
        vertices = np.concatenate((tops, bottoms), axis=0)
        return plt.Polygon(vertices, facecolor=None, fill=False)
        

class LaplaceSolver2D:

    def __init__(self, num_cells, step_size, domain_vectors=None):
        self.boundary_conditions = {}
        self.solve_time = None
        if domain_vectors:
            self._num_cells = num_cells
            self._step_size = step_size
            # solve for domain vectors
        else:
            self._domain_vectors = domain_vectors
            # solve for cells and step sizes

    def add_boundary_conditions(self, name, func, **args):
        self.boundary_conditions[name] = [func, args]

    def remove_boundary_condition(self, name):
        pass

    def solve(self, init_grid, err_lim=1E-16, step_lim=0, k=1):
        time_initial = datetime.datetime.now()
        self.result = laplace_loop(init_grid,
                                   self._num_cells[0],
                                   self._num_cells[1],

                            self.boundary_conditions, err_lim, step_lim, k)
        self.solve_time = datetime.datetime.now() - time_initial
        return self.result

    def color_plot(self, axis=None):
        pass

    def quiver_plot(self, rebin_factor=32, axis=None):
        pass


@numba.jit(nopython=True)
def temeprature_bc(grid, mask, temp):
    temp = np.ones(grid.shape)*temp
    grid = np.where(mask, temp, grid)
    return grid


@numba.jit(nopython=True)
def laplace_loop(grid, xlen, ylen, boundary_conditions,
                 err_lim=1E-16, step_lim=0, k=1):

    @numba.jit(nopython=True)
    def compute_error(old, new):
        v = (new - old).flatten()
        return np.sqrt(np.dot(v, v))

    old_grid = np.zeros(grid.shape)
    err = [1]
    steps = 0
    while err[steps] > err_lim:
        for j in range(0, xlen):
            for i in range(1, ylen - 1):
                grid[i, j] = (k * (grid[i - 1, j] +
                                   grid[i + 1, j] +
                                   grid[i, (j - 1) % xlen] +
                                   grid[i, (j + 1) % xlen]) / 4)

        # Re-assert BCs
        for bc in boundary_conditions:
            func = bc[0]
            args = bc[1]
            func(grid, **args)

        err.append(compute_error(old_grid, grid))
        old_grid = grid
        steps += 1

        if steps > step_lim and step_lim > 0:
            break

    return grid, err, steps


class BaffleOptimizer:
    
    def __init__(self, baffle_type, top_fabric_weight, bottom_fabric_weight, baffle_fabric_weight, down_fill_power) -> None:
        
        if str.lower(baffle_type) == 'rect':
            self._baffle_func = RectBaffle
        elif str.lower(baffle_type) == 'triangle':
            self._baffle_func = TriangleBaffle
        else:
            raise ValueError("Unsupported Baffle Type")
            
        self._top_fabric: float = top_fabric_weight
        self._bottom_fabric: float = bottom_fabric_weight
        self._baffle_fabric: float = baffle_fabric_weight
        self._down_fp: float = down_fill_power
            
    def width_height_analysis(self, widths, heights):
        Nw = len(widths)
        Nh = len(heights)
        
        efficiencies = np.zeros((Nw, Nh))
        thermal_res = np.zeros((Nw, Nh))
        
        for i,w in enumerate(widths):
            for j,h in enumerate(heights):
                baffle = self._baffle_func(w, h, self._top_fabric, self._bottom_fabric, self._baffle_fabric, self._down_fp)
                efficiencies[i,j] = baffle.abs_efficiency
                thermal_res[i,j] = baffle.thermal_res*baffle.width
                
        return(efficiencies, thermal_res)
    
    def summary_analysis(self, target_thermal_resistance, width_intervals, num_samples=100):
        
        summary_data = []
        efficiency_data = []

        def error(h):
            h = h[0]
            baffle = RectBaffle(w, h, self._top_fabric, self._bottom_fabric, self._baffle_fabric, self._down_fp)
            return abs(thermal_res - baffle.thermal_res*baffle.width)

        for thermal_res, (start, stop) in zip(target_thermal_resistance, width_intervals):
            ws = np.linspace(start, stop, num_samples)
            hs = []
            for w in ws:
                hs.append(fmin(error, thermal_res/2, disp=False)[0])

            efficiencies = [self._baffle_func(w, h, self._top_fabric, self._bottom_fabric, self._baffle_fabric, self._down_fp).abs_efficiency
                    for w,h in zip(ws, hs)]

            max_eff_idx = np.argmax(efficiencies)
            baffle = self._baffle_func(ws[max_eff_idx], hs[max_eff_idx], self._top_fabric,  self._bottom_fabric, self._baffle_fabric, self._down_fp)
            summary_data.append([ws[max_eff_idx], 
                                 hs[max_eff_idx], 
                                 efficiencies[max_eff_idx], 
                                 baffle.thermal_res*baffle.width, 
                                 baffle.mass/baffle.width,
                                 baffle.width])
            efficiency_data.append([ws, efficiencies])

        summary_data = np.array(summary_data).T
        names = ['w', 'h', 'efficiency', 'actual_thermal_res', 'mass', 'width']
        summary_data = pd.DataFrame({name: field for name, field in zip(names, summary_data)})
        return (summary_data, efficiency_data)
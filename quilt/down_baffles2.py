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

class RectBaffle:

    def __init__(self, width, height, top_fabric_weight, bottom_fabric_weight, baffle_fabric_weight, down_fill_power):
        self.fabric_width: float = width
        self.fabric_height: float = height
        self._top_fabric: float = convert_fabric_density(top_fabric_weight)
        self._bottom_fabric: float = convert_fabric_density(bottom_fabric_weight)
        self._baffle_fabric: float = convert_fabric_density(baffle_fabric_weight)
        self._down_density: float = convert_down_density(down_fill_power)
            
        self.catenary_param: float = self._catenary_param()
        self.width: float = 2 * self._catenary_half_width(self.catenary_param)
      
    def _catenary_param(self) -> float:
        return (minimize(self._neg_rect_baffle_area, 2).x)

    def _catenary_half_width(self, cat_param) -> float:
        return(cat_param * np.arcsinh(self.fabric_width / (2 * cat_param)))[0]

    def _catenary(self, x, cat_param):
        return cat_param * np.cosh(x / cat_param)

    def _catenary_area_under(self) -> float:
        return 2 * self.catenary_param * np.sinh(self.width/2 /
                                                  self.catenary_param)

    def _catenary_x0(self) -> float:
        return self.catenary_param * np.arcsinh(self.fabric_width /
                                                 (2 * self.catenary_param))

    def _catenary_area_inside(self, cat_param) -> float:
        return(-self.fabric_width +
               2 * self._catenary_half_width(cat_param) *
               (self._catenary(self._catenary_half_width(cat_param), cat_param) -
                (cat_param - 1)))

    def _rect_baffle_area(self, cat_param) -> float:
        return(2 * self._catenary_area_inside(cat_param) + 2 *
               self._catenary_half_width(cat_param) *
               self.fabric_height)[0]

    @property
    def area(self) -> float:
        return self._rect_baffle_area(self.catenary_param)

    def _neg_rect_baffle_area(self, cat_param) -> float:
        return -1 * self._rect_baffle_area(cat_param)

    def _catenary_y_off(self) -> float:
        return(self._catenary(self.width/2, self.catenary_param)
               - self.catenary_param)

    def _top(self, x):
        return(-self._catenary(x, self.catenary_param) +
               self._catenary_y_off() +
               self.catenary_param + self.fabric_height / 2)

    def plot(self, x=None):
        if not x:
            x = np.linspace(-self.width/2, self.width/2, 101)
        top = self._top(x)
        bottom = -top
        vertices = list(zip(x, top)) + list(zip(np.flip(x, 0), bottom))
        return plt.Polygon(vertices, facecolor=None, fill=False)

    def summary(self):
        w = self.fabric_width
        h = self.fabric_height
        print(f"Expected baffle area: {(w * h):.2f}, "
              f"actual baffle area: {self.area():.2f}, "
              f"difference: {((self.area() - w * h) / (w * h) * 100):.2f}%")
        print(f"Expected baffle width: {w} inches, "
              f"actual baffle width: {self.width:.2f}, "
              f"difference: {((self.width - w) / w * 100):.2f}%")
        print(f"Relative thermal efficiency: {(self.down_efficiency * 100):.2f}% \n")

    def _baffle_height(self, x) -> float:
        return(self.fabric_height +
               2 * (-self._catenary(x, self.catenary_param) +
                    self._catenary_y_off() +
                    self.catenary_param))

    def _inv_baffle_height(self, x) -> float:
        return 1 / self._baffle_height(x)

    @property
    def down_efficiency(self) -> float:

        thermal_res_width_normed = (self.width /
                                (quad(self._inv_baffle_height,
                                      -self.width/2,
                                      self.width/2)[0]))
        
        mass_width_normed = (1 / self.width *
                            quad(self._baffle_height,
                                 -self.width/2,
                                 self.width/2)[0])
        
        return (thermal_res_width_normed , mass_width_normed)
    
    @property
    def mass(self) -> float:
        #print(f"mass calc: fp:{self._down_density:.4f}, top:{self._top_fabric:.4f}, bottom:{self._bottom_fabric:.4f}, "
        #      f"baffle:{self._baffle_fabric:.4f}, width:{self.width:.2f}, fabric_w:{self.fabric_width:.2f}, "
        #      f"fabric_h:{self.fabric_height:.2f}")
        return (self._down_density*quad(self._baffle_height,-self.width/2,self.width/2)[0] +
                    (self._top_fabric + self._bottom_fabric) * self.fabric_width +
                    self._baffle_fabric*self.fabric_height)
    
    @property
    def thermal_res(self) -> float:
        return (1/(quad(self._inv_baffle_height,
                         -self.width/2,
                         self.width/2)[0]))
    
    @property
    def abs_efficiency(self) -> float:
        thermal_res_width_normed = (self.width * self.thermal_res)
        
        mass_width_normed = (1 / self.width * self.mass)
        
        #print(thermal_res_width_normed,mass_width_normed,self.mass)
        
        return (thermal_res_width_normed / mass_width_normed)

    def inside_mask(self, x, y):
        top = self._top(x)
        return (np.logical_and(y < top, y > -top)).T

    def above_mask(self, x, y):
        top = self._top(x)
        return (y > self._top(x)).T

    def below_mask(self, x, y):
        return (y < -self._top(x)).T

    def baffle_init_temp_dist(self, x, y, temp_lower, temp_upper):
        x0 = -self._top(x)
        x1 = self._top(x)
        y0 = temp_lower
        y1 = temp_upper
        return (y0 + (y - x0) * (y1 - y0) / (x1 - x0))


class TriangleBaffle:
    pass


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
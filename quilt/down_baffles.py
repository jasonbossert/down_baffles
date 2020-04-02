import datetime

import functools
import numba
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad

def convert_down_density(rho_imperial):
    return 1/(rho_imperial)

def convert_fabric_density(lambda_imperial):
    return lambda_imperial/36**2

class RectBaffle:

    def __init__(self, width, height, face_fabric_weight, baffle_fabric_weight, down_fill_power):
        self.fabric_width: float = width
        self.fabric_height: float = height
        self._face_fabric: float = convert_fabric_density(face_fabric_weight)
        self._baffle_fabric: float = convert_fabric_density(baffle_fabric_weight)
        self._down_density: float = convert_down_density(down_fill_power)
            
        self._catenary_param: float = self._optimized_a()
        self._half_width: float = \
            self._catenary_half_width(self._catenary_param)
        self.width: float = 2 * self._half_width
        self.down_efficiency: float = self._down_efficiency()
        self.mass: float = self._mass()
        self.thermal_res: float = self._thermal_res()
        self.abs_efficiency: float = self._absolute_efficiency()

    def _optimized_a(self) -> float:
        return (minimize(self._neg_rect_baffle_area, 2).x)

    def _catenary_half_width(self, cat_param) -> float:
        return(cat_param * np.arcsinh(self.fabric_width / (2 * cat_param)))[0]

    def _catenary(self, x, cat_param):
        return cat_param * np.cosh(x / cat_param)

    def _catenary_area_under(self) -> float:
        return 2 * self._catenary_param * np.sinh(self._half_width /
                                                  self._catenary_param)

    def _catenary_x0(self) -> float:
        return self._catenary_param * np.arcsinh(self.fabric_width /
                                                 (2 * self._catenary_param))

    def _catenary_area_inside(self, cat_param) -> float:
        return(-self.fabric_width +
               2 * self._catenary_half_width(cat_param) *
               (self._catenary(self._catenary_half_width(cat_param), cat_param) -
                (cat_param - 1)))

    def _rect_baffle_area(self, cat_param) -> float:
        return(2 * self._catenary_area_inside(cat_param) + 2 *
               self._catenary_half_width(cat_param) *
               self.fabric_height)[0]

    def area(self) -> float:
        return self._rect_baffle_area(self._catenary_param)

    def _neg_rect_baffle_area(self, cat_param) -> float:
        return -1 * self._rect_baffle_area(cat_param)

    def _catenary_y_off(self) -> float:
        return(self._catenary(self._half_width, self._catenary_param)
               - self._catenary_param)

    def _top(self, x):
        return(-self._catenary(x, self._catenary_param) +
               self._catenary_y_off() +
               self._catenary_param + self.fabric_height / 2)

    def plot(self, x=None):
        if not x:
            x = np.linspace(-self._half_width, self._half_width, 101)
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
               2 * (-self._catenary(x, self._catenary_param) +
                    self._catenary_y_off() +
                    self._catenary_param))

    def _inv_baffle_height(self, x) -> float:
        return 1 / self._baffle_height(x)

    def _down_efficiency(self) -> float:

        thermal_res_width_normed = (self.width /
                                (quad(self._inv_baffle_height,
                                      -self._half_width,
                                      self._half_width)[0]))
        
        mass_width_normed = (1 / self.width *
                            quad(self._baffle_height,
                                 -self._half_width,
                                 self._half_width)[0])
        
        return (thermal_res_width_normed , mass_width_normed)
    
    def _mass(self) -> float:
        return (self._down_density*quad(self._baffle_height,-self._half_width,self._half_width)[0] +
                    2 * self._face_fabric * self.fabric_width +
                    self._baffle_fabric*self.fabric_height)
    
    def _thermal_res(self) -> float:
        return (1/(quad(self._inv_baffle_height,
                         -self._half_width,
                         self._half_width)[0]))
    
    def _absolute_efficiency(self) -> float:
        thermal_res_width_normed = (self.width * self.thermal_res)
        
        mass_width_normed = (1 / self.width * self.mass)
        
        return (thermal_res_width_normed / mass_width_normed)
    
    def ideal_eff(self):
        return (self.fabric_height, self.fabric_height*(self._down_density + 
                       2*self._face_fabric/self.fabric_height +
                       self._baffle_fabric/self.fabric_width))
    
    def _relative_efficiency(self) -> float:
        pass
        #return (self._absolute_efficiency()/self.ideal_eff())

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


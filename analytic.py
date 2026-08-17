#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# -----------------------
# HẰNG SỐ VẬT LÍ
# -----------------------
gamma = 1.4
rho0 = 1.0              # rho / rho0
T0   = 1.0
P0   = 1.0

# -----------------------
# ỐNG THẮT VAN
# -----------------------
x_start  = 0.0
x_throat = 1.5
x_end    = 2.25



# Cho shock
exp_area     = (gamma + 1)/(2*(gamma - 1))
exp_pressure = gamma/(gamma - 1)

# -----------------------
# 2. ĐỊNH NGHĨA HÀM
# -----------------------

def A(x):
    return 1.0 + 2.2*(x - x_throat)**2

def dA_dx(x):
    return 4.4*(x - x_throat)

def solve_mach_from_area_choked(A_ratio, gamma=1.4):
    """Used when denominator is A*"""
    if abs(A_ratio - 1.0) < 1e-5 or A_ratio < 1.0:
        return 1.0, 1.0

    def f(M):
        term = (2/(gamma+1))*(1 + 0.5*(gamma-1)*M**2)
        return 1.0/M * term**((gamma+1)/(2*(gamma-1))) - A_ratio
    
    # Tìm nghiệm chắc chắn trong khoảng cận âm bằng phương pháp brentq (Không bao giờ sập)
    sol_sub = root_scalar(f, bracket=[1e-5, 0.99999], method='brentq')
    M_sub = sol_sub.root
    
    # Tìm nghiệm trong khoảng siêu âm
    sol_sup = root_scalar(f, bracket=[1.00001, 5.0], method='brentq')
    M_sup = sol_sup.root
    
    return M_sub, M_sup    

def pressure_ratio_isentropic(M, gamma=1.4):
    """Find p/p0 = f(M)"""
    return (1 + 0.5*(gamma-1)*M**2)**(-gamma/(gamma-1))

def temperature_ratio_isentropic(M, gamma=1.4):
    """Find T/T0 = f(M)"""
    return (1 + 0.5*(gamma-1)*M**2)**(-1.0)

def density_ratio_isentropic(M, gamma=1.4):
    """Find rho/rho0 = f(M)"""
    return (1 + 0.5*(gamma-1)*M**2)**(-1.0/(gamma-1))


# def solve_mach_from_area_pressure_sub(A_x, pb, gamma=1.4):
#     """Find subsonic A* then using ratio A(x)/A* to find Mach at x, can't be used if there was a normal shock"""
#     global x_end
#     f = lambda M: pressure_ratio_isentropic(M, gamma) - pb
#     M_e = float(mp.findroot(f, 0.1))
#     term = (2/(gamma+1))*(1 + 0.5*(gamma-1)*M_e**2)
#     Ae_Astar = 1.0/M_e * term**((gamma+1)/(2*(gamma-1)))
#     Ae = A(x_end)
#     A_Astar = (A_x / Ae) * Ae_Astar
#     M_sub, _ = solve_mach_from_area_choked(A_Astar, gamma=1.4)
#     return M_sub

def solve_mach_from_area_pressure_sub(A_x, pb, gamma=1.4):
    """Find subsonic Mach profile directly using mass conservation from exit plane"""
    global x_end
    
    # 1. Tính số Mach chính xác tại cửa ra từ áp suất pb
    f_exit = lambda M: pressure_ratio_isentropic(M, gamma) - pb
    Me = float(mp.findroot(f_exit, 0.1))
    
    # 2. Tính giá trị hàm diện tích-Mach tại cửa ra f(Me)
    term_e = 1 + 0.5 * (gamma - 1) * Me**2
    f_Me = (1.0 / Me) * (2 / (gamma + 1) * term_e)**exp_area
    
    # 3. Dựa vào bảo toàn lưu lượng, tính giá trị f(M) mục tiêu tại vị trí x hiện tại
    Ae = A(x_end)
    f_M_target = f_Me * (A_x / Ae)
    
    # 4. Định nghĩa hàm giải nghiệm tìm M cận âm tại vị trí x
    def f_objective(M):
        term = 1 + 0.5 * (gamma - 1) * M**2
        return (1.0 / M) * (2 / (gamma + 1) * term)**exp_area - f_M_target
        
    # Giải số tìm đúng nghiệm cận âm một cách trực tiếp, cực kỳ ổn định
    sol = root_scalar(f_objective, bracket=[1e-5, 0.99999], method='brentq')
    return sol.root



# Biến toàn cục phục vụ hàm shock
pb_shock = None
P02      = None
Ae_Astar = None

def solve_shock_mach_exit_formula(Me):
    term = 1 + 0.5*(gamma-1)*Me**2
    lhs  = ( (2*term / (gamma+1))**exp_area ) / (Ae_Astar * Me)
    rhs  = pb_shock * term**exp_pressure
    return lhs - rhs

def solve_shock_mach_beforeshock_formula(M1):
    p2_p1 = (2*gamma*M1**2 - (gamma-1)) / (gamma+1)
    M2_sq = (1 + 0.5*(gamma-1)*M1**2) / (gamma*M1**2 - 0.5*(gamma-1))
    M2    = mp.sqrt(M2_sq)
    term1 = p2_p1
    term2 = (1 + 0.5*(gamma-1)*M2**2)**exp_pressure
    term3 = (1 + 0.5*(gamma-1)*M1**2)**exp_pressure
    return term1 * term2 / term3 - P02/P0

def solve_shock_area_formula(M):
    term = 1 + 0.5*(gamma-1)*M**2
    return 1.0/M * (2/(gamma+1)*term)**exp_area

def compute_analytic(pb, x_np):
    """main function"""
    global pb_shock, P02, Ae_Astar

    A_throat  = A(x_throat)
    Ae_Astar  = A(x_end) / A_throat

    # Mach thiết kế ở cửa ra (isentropic 2 nhánh)
    M_sub, M_sup = solve_mach_from_area_choked(Ae_Astar, gamma)
    P_sub = pressure_ratio_isentropic(M_sub, gamma)
    P_sup = pressure_ratio_isentropic(M_sup, gamma)
    P_shock_exit = (1 + (2 * gamma / (gamma + 1)) * (M_sup**2 - 1)) * P_sup

    x_shock = None
    r02_01  = 1.0   # p02/p01, mặc định =1 nếu không có shock

    # --- Trường hợp có shock trong phần ống phân kỳ ---
    if P_shock_exit < pb < P_sub:
        pb_shock = pb
        # Mach exit sau shock
        M_e_shock = float(mp.findroot(solve_shock_mach_exit_formula, 0.5))
        # Tổng áp downstream P02
        P02 = pb * (1 + 0.5*(gamma-1)*M_e_shock**2)**exp_pressure
        # Mach upstream shock
        M1 = float(mp.findroot(solve_shock_mach_beforeshock_formula, 2.0))
        A1 = A_throat * solve_shock_area_formula(M1)
        x_shock = np.sqrt((A1 - 1)/2.2) + x_throat
        # Tỉ số tổng áp qua shock
        termA = ((gamma + 1.0)*M1**2) / ((gamma - 1.0)*M1**2 + 2.0)
        termB = (gamma + 1.0) / (2.0*gamma*M1**2 - (gamma - 1.0))
        r02_01 = (termA**(gamma/(gamma - 1.0))) * (termB**(1.0/(gamma - 1.0)))

    # --- Tính Mach profile ---
    A_star_2 = A_throat / r02_01  
    M_cal = np.zeros_like(x_np)
    for i, xx in enumerate(x_np):
        A_r = A(xx)/A_throat
        if pb <= P_shock_exit:   # supersonic toàn phần
            if i == 0: print("--- Đang chạy nhánh 1: Siêu âm toàn phần trong ống ---")
            if xx <= x_throat:
                M_cal[i], _ = solve_mach_from_area_choked(A_r, gamma)
            else:
                _, M_cal[i] = solve_mach_from_area_choked(A_r, gamma)
        elif pb >= P_sub: # subsonic toàn phần
            if i == 0: print("--- Đang chạy nhánh 2: Cận âm toàn phần ---")
            M_cal[i] = solve_mach_from_area_pressure_sub(A(xx), pb, gamma)
        else:             # có shock
            if i == 0: print("--- Đang chạy nhánh 3: Có Shock ---") 
            if xx <= x_throat:
                M_cal[i], _ = solve_mach_from_area_choked(A_r, gamma)
            elif xx <= x_shock:
                _, M_cal[i] = solve_mach_from_area_choked(A_r, gamma)
            else:
                A_r_shock = A(xx) / A_star_2
                M_cal[i], _ = solve_mach_from_area_choked(A_r_shock, gamma) # Lấy nghiệm cận âm (M_sub)

    # --- P, T, rho ---
    P_cal   = pressure_ratio_isentropic(M_cal, gamma) * P0
    T_cal   = temperature_ratio_isentropic(M_cal, gamma) * T0
    rho_cal = density_ratio_isentropic(M_cal, gamma) * rho0

    if x_shock is not None:
        mask = (x_np > x_shock)
        P_cal[mask]   *= r02_01
        rho_cal[mask] *= r02_01   # vì rho0 ∝ p0; T0 giữ nguyên

    return M_cal, P_cal, T_cal, rho_cal


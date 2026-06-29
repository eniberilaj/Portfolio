"""
Spatial Xenon-135 and Iodine-135 transient solver.
Simulates the 'Iodine Pit' and spatial Xenon oscillations over a 72-hour window.
"""
import numpy as np
from app.physics.diffusion import solve_core, CoreParams

# Decay constants (s^-1) and yield fractions
LAMBDA_I  = 2.87e-5
LAMBDA_XE = 2.09e-5
GAMMA_I   = 0.0639
GAMMA_XE  = 0.00237
SIGMA_A_XE = 2.65e-18  # cm^2
SIGMA_F    = 0.10      # cm^-1 (macroscopic fission)
PHI_NOMINAL = 3.0e13   # n/cm^2/s (nominal full power flux)

def simulate_transient(state: dict, scenario="scram", duration_h=72):
    hours = int(duration_h)
    
    # 1. Establish Equilibrium State (100% Power)
    p = CoreParams(
        rod_insertion=state.get("rod_insertion", 0.22),
        enrichment=state.get("enrichment", 3.2),
        power_demand=1.0, 
        refine=2
    )
    base_solve = solve_core(p)
    flux_shape = np.array(base_solve["power_map"]) 
    phi = flux_shape * PHI_NOMINAL
    
    # Equilibrium concentrations: dI/dt = 0, dXe/dt = 0
    I_map = (GAMMA_I * SIGMA_F * phi) / LAMBDA_I
    Xe_map = ((GAMMA_I + GAMMA_XE) * SIGMA_F * phi) / (LAMBDA_XE + SIGMA_A_XE * phi)
    
    snapshots = []
    
    # 2. Apply Transient Scenario
    if scenario == "scram":
        p.power_demand = 0.0
        # Simulating control rods dropping in
        p.rod_insertion = 1.0 
    elif scenario == "oscillation":
        # Simulating an asymmetric rod drop on one side to trigger a spatial flux tilt
        p.rod_insertion = 0.22
        # We will manually tilt the flux in the loop to simulate a rod stuck on one side
    
    # 3. Integrate 72 hours (Inner step = 5 minutes for stability)
    dt_s = 300.0 
    steps_per_hour = int(3600 / dt_s)
    
    for h in range(hours + 1):
        # Save snapshot every hour
        snapshots.append({
            "hour": h,
            "xe_map": np.round(Xe_map, 2).tolist(),
            "power_map": np.round(flux_shape, 3).tolist(),
            "xe_max": float(Xe_map.max())
        })
        
        # Advance 1 hour
        for _ in range(steps_per_hour):
            # Recalculate Flux based on current Xenon poisoning
            p.xe_map = Xe_map
            core = solve_core(p)
            flux_shape = np.array(core["power_map"])
            
            if scenario == "oscillation" and h < 2:
                # Force a temporary flux tilt for the first 2 hours to start the swing
                tilt = np.linspace(0.5, 1.5, 45)
                flux_shape = flux_shape * tilt
                
            phi = flux_shape * PHI_NOMINAL * p.power_demand
            
            # Bateman ODEs (Euler step)
            dI = (GAMMA_I * SIGMA_F * phi) - (LAMBDA_I * I_map)
            dXe = (GAMMA_XE * SIGMA_F * phi) + (LAMBDA_I * I_map) - (LAMBDA_XE * Xe_map) - (SIGMA_A_XE * Xe_map * phi)
            
            I_map += dI * dt_s
            Xe_map += dXe * dt_s

    return {
        "scenario": scenario,
        "duration_h": duration_h,
        "frames": snapshots
    }
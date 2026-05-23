STEFAN_BOLTZMANN = 5.67e-8  # W/m2-K4
HOURS_PER_YEAR = 8760

MATERIALS = {
    "concrete": {"conductivity": 1.13, "density": 2000, "specific_heat": 1000},
    "cast_concrete": {"conductivity": 1.40, "density": 2100, "specific_heat": 840},
    "concrete_block": {"conductivity": 0.51, "density": 1400, "specific_heat": 1000},
    "brick": {"conductivity": 0.84, "density": 1700, "specific_heat": 800},
    "dense_plaster": {"conductivity": 0.50, "density": 1300, "specific_heat": 1000},
    "polyurethane_insulation": {"conductivity": 0.025, "density": 30, "specific_heat": 1400},
    "screed": {"conductivity": 0.41, "density": 1200, "specific_heat": 840},
    "soil_subsoil": {"conductivity": 1.50, "density": 1900, "specific_heat": 1000},
    "soil_topsoil": {"conductivity": 0.80, "density": 1300, "specific_heat": 1000},
    "sod": {"conductivity": 0.30, "density": 800, "specific_heat": 1800},
    "sand": {"conductivity": 1.74, "density": 1700, "specific_heat": 800},
    "air_gap_25mm": {"conductivity": 0.18, "density": 1.2, "specific_heat": 1000},
    "waterproof_membrane": {"conductivity": 0.23, "density": 1100, "specific_heat": 1000},
}

SURFACE_ABSORPTIVITY = {
    "concrete": 0.65,
    "brick": 0.70,
    "grass": 0.75,
    "roof_membrane": 0.85,
    "dark_roof": 0.90,
}

SURFACE_EMISSIVITY = {
    "concrete": 0.90,
    "brick": 0.90,
    "grass": 0.95,
    "roof_membrane": 0.90,
}

R_SI_WALL = 0.13  # m2-K/W, interior wall surface resistance
R_SO = 0.04  # m2-K/W, exterior surface resistance
R_SI_GROUND = 0.00  # ground contact to deep soil (handled by layer depth)

def get_emission_factors(speed_kmh, vehicle_type, config):
    """
    Returns emission factors for the given speed and vehicle type.
    """

    bins = config["vehicle_types"][vehicle_type]["speed_bins"]

    for min_v, max_v, co2, nox in bins:
        if min_v <= speed_kmh < max_v:
            return {
                "co2": co2,
                "nox": nox
            }

    return {
        "co2": 0,
        "nox": 0
    }
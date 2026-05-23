from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


@dataclass
class WeatherData:
    temp_air: np.ndarray  # degC, (8760,)
    dew_point: np.ndarray  # degC
    ghi: np.ndarray  # W/m2
    dni: np.ndarray  # W/m2
    dhi: np.ndarray  # W/m2
    wind_speed: np.ndarray  # m/s
    infrared_horizontal: np.ndarray  # W/m2, horizontal infrared radiation from sky
    latitude: float
    longitude: float
    altitude: float
    tz: float
    location: pvlib.location.Location
    times: pd.DatetimeIndex

    @property
    def solar_position(self) -> pd.DataFrame:
        if not hasattr(self, "_solar_position"):
            self._solar_position = self.location.get_solarposition(self.times)
        return self._solar_position

    @property
    def annual_mean_temp(self) -> float:
        return float(np.mean(self.temp_air))


def load_epw(path: str | Path) -> WeatherData:
    df, meta = pvlib.iotools.read_epw(str(path))

    location = pvlib.location.Location(
        latitude=meta["latitude"],
        longitude=meta["longitude"],
        altitude=meta["altitude"],
        tz=meta["TZ"],
    )

    return WeatherData(
        temp_air=df["temp_air"].values,
        dew_point=df["temp_dew"].values,
        ghi=df["ghi"].values,
        dni=df["dni"].values,
        dhi=df["dhi"].values,
        wind_speed=df["wind_speed"].values,
        infrared_horizontal=df["ghi_infrared"].values,
        latitude=meta["latitude"],
        longitude=meta["longitude"],
        altitude=meta["altitude"],
        tz=meta["TZ"],
        location=location,
        times=df.index,
    )

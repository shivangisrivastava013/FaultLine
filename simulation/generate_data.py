"""
Synthetic Delivery Data Generator for FaultLine.

Generates realistic delivery trips with:
- distance_km (continuous)
- traffic_level ('low', 'medium', 'high', 'jam')
- weather ('clear', 'rain', 'storm', 'snow')
- hour_of_day (0-23)
- restaurant_load (0.0 - 1.0)
- delivery_duration_minutes (ground truth target)

Includes deterministic random seeds and partitions for train, val, and test.
"""

from __future__ import annotations

import os
import uuid
import datetime
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


# Documented physical parameters
TRAFFIC_BASE_SPEED_KMH: Dict[str, float] = {
    "low": 42.0,
    "medium": 28.0,
    "high": 17.0,
    "jam": 9.0,
}

WEATHER_SPEED_MULTIPLIER: Dict[str, float] = {
    "clear": 1.00,
    "rain": 0.82,
    "storm": 0.65,
    "snow": 0.58,
}

VALID_TRAFFIC_LEVELS = list(TRAFFIC_BASE_SPEED_KMH.keys())
VALID_WEATHER_CONDITIONS = list(WEATHER_SPEED_MULTIPLIER.keys())


def calculate_delivery_duration(
    distance_km: float | np.ndarray,
    traffic_level: str | np.ndarray,
    weather: str | np.ndarray,
    restaurant_load: float | np.ndarray,
    noise_std: float = 2.5,
    rng: Optional[np.random.Generator] = None,
) -> float | np.ndarray:
    """
    Computes ground truth delivery duration in minutes based on documented physics:
    prep_time = 12.0 + (16.0 * restaurant_load)
    effective_speed = traffic_speed * weather_multiplier
    transit_time = (distance_km / effective_speed) * 60
    total = prep_time + transit_time + noise
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if isinstance(traffic_level, str):
        speed_base = TRAFFIC_BASE_SPEED_KMH.get(traffic_level, 25.0)
        weather_factor = WEATHER_SPEED_MULTIPLIER.get(weather, 0.9)  # type: ignore[arg-type]
        effective_speed = max(speed_base * weather_factor, 5.0)
        prep = 12.0 + (16.0 * float(restaurant_load))
        transit = (float(distance_km) / effective_speed) * 60.0
        noise = rng.normal(0, noise_std)
        return max(8.0, round(float(prep + transit + noise), 2))

    # Array vectorization
    n = len(distance_km)
    speed_base = np.array([TRAFFIC_BASE_SPEED_KMH.get(t, 25.0) for t in traffic_level])
    weather_factor = np.array([WEATHER_SPEED_MULTIPLIER.get(w, 0.9) for w in weather])
    effective_speed = np.maximum(speed_base * weather_factor, 5.0)
    prep = 12.0 + (16.0 * np.asarray(restaurant_load, dtype=float))
    transit = (np.asarray(distance_km, dtype=float) / effective_speed) * 60.0
    noise = rng.normal(0, noise_std, size=n)
    total = np.maximum(8.0, prep + transit + noise)
    return np.round(total, 2)


def generate_delivery_records(
    num_samples: int = 10000,
    seed: int = 42,
    start_time: Optional[datetime.datetime] = None,
    time_step_seconds: int = 30,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic delivery dataset with timestamps, event IDs,
    correlated features (e.g. rush hours correlate with high traffic and restaurant load).
    """
    rng = np.random.default_rng(seed)

    if start_time is None:
        start_time = datetime.datetime(2026, 1, 15, 8, 0, 0, tzinfo=datetime.timezone.utc)

    # 1. Timestamps & hour of day
    timestamps = [start_time + datetime.timedelta(seconds=i * time_step_seconds) for i in range(num_samples)]
    hours = np.array([ts.hour for ts in timestamps])

    # 2. Distance in km (gamma distribution skewed towards 2-8 km, max ~22km)
    distances = np.clip(rng.gamma(shape=2.5, scale=2.2, size=num_samples), 0.5, 24.0)
    distances = np.round(distances, 2)

    # 3. Weather probabilities: clear (65%), rain (22%), storm (7%), snow (6%)
    weather_choices = rng.choice(
        VALID_WEATHER_CONDITIONS,
        size=num_samples,
        p=[0.65, 0.22, 0.07, 0.06]
    )

    # 4. Traffic & Restaurant Load correlated with meal times (12-14, 18-21)
    traffic_levels = []
    restaurant_loads = []

    for h in hours:
        is_rush = (12 <= h <= 14) or (18 <= h <= 21)
        if is_rush:
            p_traffic = [0.05, 0.25, 0.45, 0.25]  # more high and jam
            load_base = rng.uniform(0.55, 0.98)
        elif (23 <= h or h <= 5):
            p_traffic = [0.75, 0.20, 0.04, 0.01]  # mostly low
            load_base = rng.uniform(0.05, 0.35)
        else:
            p_traffic = [0.35, 0.45, 0.16, 0.04]  # moderate
            load_base = rng.uniform(0.25, 0.65)

        t_level = rng.choice(VALID_TRAFFIC_LEVELS, p=p_traffic)
        traffic_levels.append(t_level)
        restaurant_loads.append(round(float(np.clip(load_base, 0.0, 1.0)), 2))

    traffic_levels = np.array(traffic_levels)
    restaurant_loads = np.array(restaurant_loads)

    # 5. Durations
    durations = calculate_delivery_duration(
        distance_km=distances,
        traffic_level=traffic_levels,
        weather=weather_choices,
        restaurant_load=restaurant_loads,
        noise_std=2.5,
        rng=rng,
    )

    # 6. Event IDs
    event_ids = [f"evt_{rng.integers(0, 2**60):012x}" for _ in range(num_samples)]

    df = pd.DataFrame({
        "event_id": event_ids,
        "timestamp": [ts.isoformat() for ts in timestamps],
        "distance_km": distances,
        "traffic_level": traffic_levels,
        "weather": weather_choices,
        "hour_of_day": hours,
        "restaurant_load": restaurant_loads,
        "delivery_duration_minutes": durations,
    })

    return df


def generate_and_save_dataset_splits(
    output_dir: str = "data",
    num_samples: int = 15000,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates dataset and writes train.parquet, val.parquet, test.parquet,
    plus a reference_baseline.parquet used for data drift monitoring.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = generate_delivery_records(num_samples=num_samples, seed=seed)

    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * (train_ratio + val_ratio))

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_val].copy()
    test_df = df.iloc[n_val:].copy()

    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    # Also save reference dataset for data drift monitoring (from validation split)
    val_df.to_parquet(os.path.join(output_dir, "reference_baseline.parquet"), index=False)
    print(f"Saved dataset splits to {output_dir}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    generate_and_save_dataset_splits()

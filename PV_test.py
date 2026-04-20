# PV_test.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from Assets import PVAsset

# 0) Output folder
FIG_DIR = Path("Figures")
FIG_DIR.mkdir(exist_ok=True)

# 1) Load Renewables.ninja PV CSV (corrected file)
PV_FILE = Path("Data") / "ninja_pv_52.9534_-1.1496_corrected.csv"
df = pd.read_csv(PV_FILE, skiprows=3, usecols=[0, 1, 2, 3, 4, 5])

df.columns = ["time", "local_time", "electricity_kw", "irr_dir", "irr_dif", "temp_c"]

# Parse time column properly
raw_time = df["time"].astype(str).str.strip()
df["time"] = pd.to_datetime(
    raw_time,
    format="%d/%m/%Y %H:%M",
    dayfirst=True,
    errors="coerce"
)

if df["time"].isna().any():
    bad = raw_time[df["time"].isna()].head(5)
    raise ValueError(f"Some time values could not be parsed. Examples:\n{bad}")

# Set real datetime index
df = df.set_index("time").sort_index()

# Convert numeric columns safely
for col in ["electricity_kw", "irr_dir", "irr_dif", "temp_c"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing core inputs
df = df.dropna(subset=["electricity_kw", "irr_dir", "irr_dif", "temp_c"])

# GHI approximation (W/m^2)
df["ghi_wm2"] = df["irr_dir"] + df["irr_dif"]

# If irradiance is in kW/m^2, convert to W/m^2
if df["ghi_wm2"].max() <= 2.0:
    df["ghi_wm2"] *= 1000.0

ghi = df["ghi_wm2"].to_numpy(dtype=float)
temp = df["temp_c"].to_numpy(dtype=float)

# 2) Build PV model
T = len(df)
dt = 1.0
dt_ems = 1.0
T_ems = T

rated_power_kw = 3.5
bus_id = 1
PR = 0.79

pv = PVAsset(
    rated_power_kw=rated_power_kw,
    bus_id=bus_id,
    dt=dt,
    T=T,
    dt_ems=dt_ems,
    T_ems=T_ems,
    gamma_per_c=-0.004,
    pr=PR,
)

# 3) Generate model output
model_kw = np.asarray(pv.generate(ghi, temp), dtype=float)
ninja_kw = df["electricity_kw"].to_numpy(dtype=float)

# 4) Validation metrics
mask = np.isfinite(model_kw) & np.isfinite(ninja_kw)

df_valid = df.iloc[mask].copy().sort_index()
model_kw = model_kw[mask]
ninja_kw = ninja_kw[mask]

df_valid["model_kw"] = model_kw
df_valid["ninja_kw"] = ninja_kw

rmse = np.sqrt(np.mean((model_kw - ninja_kw) ** 2))
mae = np.mean(np.abs(model_kw - ninja_kw))

ss_res = np.sum((ninja_kw - model_kw) ** 2)
ss_tot = np.sum((ninja_kw - np.mean(ninja_kw)) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

E_ninja_kwh = np.sum(ninja_kw) * dt
E_model_kwh = np.sum(model_kw) * dt
energy_err_pct = (
    100.0 * (E_model_kwh - E_ninja_kwh) / E_ninja_kwh
    if E_ninja_kwh != 0 else np.nan
)

print("\n--- PV VALIDATION ---")
print(f"Rated power (kW): {rated_power_kw}")
print(f"PR: {PR}")
print(f"RMSE (kW): {rmse:.4f}")
print(f"MAE  (kW): {mae:.4f}")
print(f"R^2: {r2:.4f}")
print(f"Energy model (kWh): {E_model_kwh:.2f}")
print(f"Energy ref   (kWh): {E_ninja_kwh:.2f}")
print(f"Energy error (%): {energy_err_pct:.2f}%")

# 5) Plots
param_text = (
    f"$P_{{rated}}$ = {rated_power_kw:.2f} kW\n"
    f"$\\gamma$ = {pv.gamma_per_c:.4f} /°C, PR = {pv.pr:.2f}\n"
    f"$R^2$ = {r2:.3f}, Energy error = {energy_err_pct:.2f}%"
)

# (A) Power vs GHI
ghi_arr = df_valid["ghi_wm2"].to_numpy(dtype=float)
idx = np.argsort(ghi_arr)

plt.figure(figsize=(9, 5))
plt.scatter(ghi_arr[idx], model_kw[idx], s=8, label="Model")
plt.scatter(ghi_arr[idx], ninja_kw[idx], s=8, label="Ninja ref")
plt.xlabel("GHI (W/m$^2$)")
plt.ylabel("Power (kW)")
plt.title("PVAsset Validation: Power vs GHI")
plt.grid(True)
plt.legend()

plt.gca().text(
    0.60,
    0.20,
    param_text,
    transform=plt.gca().transAxes,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

plt.tight_layout()
plt.savefig(str(FIG_DIR / f"PV_power_vs_GHI_{rated_power_kw:.1f}kW.png"), dpi=150)
plt.close()

# (B) Time series (first 7 days)
days = 7
samples = int(days * 24 / dt)

plt.figure(figsize=(10, 5))
plt.plot(
    df_valid.index[:samples],
    df_valid["ninja_kw"].to_numpy()[:samples],
    label="Ninja ref (kW)"
)
plt.plot(
    df_valid.index[:samples],
    df_valid["model_kw"].to_numpy()[:samples],
    label="Model (kW)"
)
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title(f"PVAsset Validation: Model vs Ninja (first {days} days)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(str(FIG_DIR / f"PV_timeseries_first{days}days_{rated_power_kw:.1f}kW.png"), dpi=150)
plt.close()

# (C) Daily average comparison
daily_pv = df_valid[["model_kw", "ninja_kw"]].resample("D").mean()

plt.figure(figsize=(10, 5))
plt.plot(daily_pv.index, daily_pv["model_kw"], label="Model (daily average)")
plt.plot(daily_pv.index, daily_pv["ninja_kw"], label="Ninja ref (daily average)")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("PVAsset Validation: Daily Average Power Comparison")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(str(FIG_DIR / f"PV_daily_average_{rated_power_kw:.1f}kW.png"), dpi=150)
plt.close()

# (D) 15-day annual trend
pv_15day = df_valid[["model_kw", "ninja_kw"]].resample("15D").mean()

plt.figure(figsize=(10, 5))
plt.plot(pv_15day.index, pv_15day["model_kw"], linestyle="--", linewidth=1.8, label="Model")
plt.plot(pv_15day.index, pv_15day["ninja_kw"], linewidth=1.8, label="Ninja")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("PVAsset Validation: 15-day Annual Trend")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(str(FIG_DIR / f"PV_15day_trend_{rated_power_kw:.1f}kW.png"), dpi=150)
plt.close()

print(f"\nPlots saved to: {FIG_DIR.resolve()}")
print("Done")
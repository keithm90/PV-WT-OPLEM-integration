# WT_test.py
from pathlib import Path

FIG_DIR = Path("Figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Assets import WTAsset
from Data.load_ninja import load_ninja_wind


# -------------------------
# 1) Load Renewables.ninja WIND CSV
# -------------------------
WT_FILE = Path("Data") / "ninja_wind_52.9534_-1.1496_corrected.csv"
weather = load_ninja_wind(str(WT_FILE))   # returns index=time + columns like wind_ms, ninja_electricity

# Choose wind-speed column safely
wind_col = "wind_ms" if "wind_ms" in weather.columns else "wind_speed"

# -------------------------
# 2) Build WTAsset instance
# -------------------------
rated_power_kw = 7580.0   # 7.58 MW turbine
bus_id = 1

T = len(weather)
dt = 1.0
dt_ems = dt
T_ems = T

wt = WTAsset(
    rated_power_kw,
    bus_id,
    dt,
    T,
    dt_ems,
    T_ems,
    v_rated=12.0
)

# -------------------------
# 3) Generate wind power profile (kW)
# -------------------------
p_model_kw = wt.generate(weather[wind_col])

if not isinstance(p_model_kw, np.ndarray):
    p_model_kw = np.asarray(p_model_kw, dtype=float)

p_model_kw = np.nan_to_num(p_model_kw, nan=0.0)
p_model_kw = np.clip(p_model_kw, 0.0, rated_power_kw)

# -------------------------
# 4) Build reference from Ninja electricity column
# -------------------------
ninja_e = weather["ninja_electricity"].to_numpy(dtype=float)
ninja_e = np.nan_to_num(ninja_e, nan=0.0)
ninja_e = np.clip(ninja_e, 0.0, None)

# If Ninja output looks normalised, scale by rated power
if np.max(ninja_e) <= 2.0:
    p_ref_kw = ninja_e * rated_power_kw
else:
    p_ref_kw = ninja_e

# -------------------------
# 5) Validation metrics
# -------------------------
err = p_model_kw - p_ref_kw
rmse = float(np.sqrt(np.mean(err ** 2)))
mae = float(np.mean(np.abs(err)))

energy_model = float(np.sum(p_model_kw))
energy_ref = float(np.sum(p_ref_kw))
energy_error_pct = float(100.0 * (energy_model - energy_ref) / (energy_ref + 1e-9))

cf_model = p_model_kw / rated_power_kw
cf_ref = p_ref_kw / rated_power_kw

ss_res = float(np.sum(err ** 2))
ss_tot = float(np.sum((p_ref_kw - np.mean(p_ref_kw)) ** 2))
r2 = float(1.0 - ss_res / (ss_tot + 1e-9))

print("\n--- WIND VALIDATION ---")
print(f"Rated power (kW): {rated_power_kw}")
print(f"RMSE (kW): {rmse:.4f}")
print(f"MAE  (kW): {mae:.4f}")
print(f"Energy model (kWh if hourly): {energy_model:.3f}")
print(f"Energy ref   (kWh if hourly): {energy_ref:.3f}")
print(f"Energy error (%): {energy_error_pct:.2f}%")
print(f"R^2: {r2:.4f}")
print(f"Model max (kW): {np.max(p_model_kw):.3f} | Ref max (kW): {np.max(p_ref_kw):.3f}")
print(f"Wind min/max (m/s): {float(weather[wind_col].min()):.3f} / {float(weather[wind_col].max()):.3f}")

# -------------------------
# 6) Quick comparison plots
# -------------------------
plt.figure()
plt.plot(p_model_kw, linestyle="--", label="Model")
plt.plot(p_ref_kw, label="Ninja")
plt.legend()
plt.savefig(str(FIG_DIR / "wt_test.png"), dpi=150)
plt.close()

plt.figure()
plt.plot(p_model_kw[0:24], linestyle="--", label="Model")
plt.plot(p_ref_kw[0:24], label="Ninja")
plt.legend()
plt.savefig(str(FIG_DIR / "wt_test_day.png"), dpi=150)
plt.close()

# -------------------------
# 7) Detailed plots
# -------------------------
wind_ms = weather[wind_col].to_numpy(dtype=float)

# Safe metrics for annotation
mask = np.isfinite(p_model_kw) & np.isfinite(p_ref_kw)
if np.any(mask):
    ss_res_val = np.sum((p_ref_kw[mask] - p_model_kw[mask]) ** 2)
    ss_tot_val = np.sum((p_ref_kw[mask] - np.mean(p_ref_kw[mask])) ** 2)
    r2_val = 1.0 - (ss_res_val / ss_tot_val) if ss_tot_val > 0 else np.nan

    E_model = np.sum(p_model_kw[mask]) * dt
    E_ref = np.sum(p_ref_kw[mask]) * dt
    energy_err_pct_val = 100.0 * (E_model - E_ref) / E_ref if E_ref != 0 else np.nan
else:
    r2_val = np.nan
    energy_err_pct_val = np.nan

param_text = (
    f"$P_{{rated}}$ = {wt.rated_power_kw:.2f} kW\n"
    f"$v_{{ci}}$ = {wt.v_cut_in:.1f} m/s, "
    f"$v_{{r}}$ = {wt.v_rated:.1f} m/s, "
    f"$v_{{co}}$ = {wt.v_cut_out:.1f} m/s\n"
    f"$R^2$ = {r2_val:.3f}, Energy error = {energy_err_pct_val:.2f}%"
)

# (A) Power vs wind speed
v_grid = np.linspace(0.0, 30.0, 400)
p_curve = np.zeros_like(v_grid, dtype=float)

ramp = (v_grid >= wt.v_cut_in) & (v_grid < wt.v_rated)
denom = wt.v_rated**3 - wt.v_cut_in**3
if denom > 0:
    p_curve[ramp] = wt.rated_power_kw * (
        (v_grid[ramp]**3 - wt.v_cut_in**3) / denom
    )

rated = (v_grid >= wt.v_rated) & (v_grid <= wt.v_cut_out)
p_curve[rated] = wt.rated_power_kw
p_curve = np.clip(p_curve, 0.0, wt.rated_power_kw)

plt.figure(figsize=(9, 5))

mask_plot = v_grid <= wt.v_cut_out
plt.plot(v_grid[mask_plot], p_curve[mask_plot], label="Model curve (≤ cut-out)")
plt.plot([wt.v_cut_out], [0.0], marker="x", markersize=8, label="Cut-out")

step = 8
plt.scatter(wind_ms[::step], p_model_kw[::step], s=10, label="Model (data)")
plt.scatter(wind_ms[::step], p_ref_kw[::step], s=10, label="Ninja ref (data)")

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power (kW)")
plt.title("WTAsset Validation: Power vs Wind Speed")
plt.grid(True)
plt.axvline(wt.v_cut_in, linestyle="--", linewidth=1, alpha=0.6)
plt.axvline(wt.v_rated, linestyle="--", linewidth=1, alpha=0.6)
plt.axvline(wt.v_cut_out, linestyle="--", linewidth=1, alpha=0.6)

y_top = plt.ylim()[1]
plt.text(wt.v_cut_in + 0.2, 0.95 * y_top, "cut-in", rotation=90, va="top")
plt.text(wt.v_rated + 0.2, 0.95 * y_top, "rated", rotation=90, va="top")
plt.text(wt.v_cut_out + 0.2, 0.95 * y_top, "cut-out", rotation=90, va="top")

plt.xlim(0, 30)
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
plt.savefig(str(FIG_DIR / f"WT_power_vs_windspeed_{rated_power_kw:.1f}kW.png"), dpi=150)
plt.close()

# (B) Time series comparison
n = min(300, len(weather))
plt.figure(figsize=(10, 5))
plt.plot(weather.index[:n], p_model_kw[:n], label="Model (kW)")
plt.plot(weather.index[:n], p_ref_kw[:n], label="Ninja ref (kW)")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("WTAsset Validation: Model vs Ninja (first 300 timesteps)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(str(FIG_DIR / f"WT_timeseries_first300_{rated_power_kw:.1f}kW.png"), dpi=150)
plt.close()

# (C) Daily average power comparison
daily_model = (
    weather.assign(model_kw=p_model_kw, ref_kw=p_ref_kw)
    .resample("D")[["model_kw", "ref_kw"]]
    .mean()
)

plt.figure(figsize=(10, 5))
plt.plot(daily_model.index, daily_model["model_kw"], label="Model (daily average)")
plt.plot(daily_model.index, daily_model["ref_kw"], label="Ninja ref (daily average)")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("WTAsset Validation: Daily Average Power Comparison")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(str(FIG_DIR / f"WT_daily_average_{rated_power_kw:.1f}kW.png"), dpi=150)
plt.close()

# (D) Smoothed annual trend plot using 15-day averages
smooth_df = (
    weather.assign(model_kw=p_model_kw, ref_kw=p_ref_kw)
    .resample("15D")[["model_kw", "ref_kw"]]
    .mean()
)

plt.figure(figsize=(10, 5))
plt.plot(smooth_df.index, smooth_df["model_kw"], linestyle="--", linewidth=1.8, label="Model")
plt.plot(smooth_df.index, smooth_df["ref_kw"], linewidth=1.8, label="Ninja")

plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("WTAsset Validation: 15 day Annual Trend")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(str(FIG_DIR / f"WT_smoothed_15day_dates_{rated_power_kw:.1f}kW.png"), dpi=150)
plt.close()

print(f"\nPlots saved to: {FIG_DIR.resolve()}")
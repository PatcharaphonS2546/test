from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Optional, Tuple, List
from numba import njit

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

@dataclass
class Sample:
    ts_ms: float
    latency_ms: float | None
    pred_px: tuple[int,int] | None
    target_px: tuple[int,int] | None
    point_id: int | None

class MetricsRecorder:
    def __init__(self):
        self.samples: list[Sample] = []
        self._maxlen = 120000
        self._df_cache = None
        self._df_cache_count = 0

    def record(self, *, latency_ms, pred_px, target_px=None, point_id=None):
        ts_ms = time.time() * 1000.0
        if len(self.samples) >= self._maxlen:
            self.samples = self.samples[-self._maxlen//2:]
        self.samples.append(Sample(ts_ms, latency_ms, pred_px, target_px, point_id))
        self._df_cache = None
        self._df_cache_count = len(self.samples)

    def _df(self):
        if not self.samples or pd is None:
            return None
        # Use cache if available and sample count unchanged
        if self._df_cache is not None and self._df_cache_count == len(self.samples):
            return self._df_cache
        pred_arr = np.array([s.pred_px if s.pred_px is not None else (np.nan, np.nan) for s in self.samples])
        tgt_arr = np.array([s.target_px if s.target_px is not None else (np.nan, np.nan) for s in self.samples])
        err_arr = np.linalg.norm(pred_arr - tgt_arr, axis=1)
        rows = [
            dict(ts_ms=s.ts_ms,
                 latency_ms=s.latency_ms,
                 pred_x=s.pred_px[0] if s.pred_px else None,
                 pred_y=s.pred_px[1] if s.pred_px else None,
                 tgt_x=s.target_px[0] if s.target_px else None,
                 tgt_y=s.target_px[1] if s.target_px else None,
                 point_id=s.point_id,
                 err_px=err if not np.isnan(err) else None)
            for s, err in zip(self.samples, err_arr)
        ]
        df = pd.DataFrame(rows)
        self._df_cache = df
        self._df_cache_count = len(self.samples)
        return df

    def summarize(self):
        if pd is None:
            return None, None
        df = self._df()
        if df is None or df.empty:
            return (
                pd.DataFrame({"metric":["Latency(ms)","Jitter(px)","MAE(px)","RMSE(px)"], "value":[np.nan]*4}),
                pd.DataFrame(columns=["point_id","jitter_px","mae_px","rmse_px","n"])
            )
        df_t = df.dropna(subset=["tgt_x","tgt_y","err_px"])
        lat_mean = float(np.nanmean(df["latency_ms"])) if "latency_ms" in df and len(df) else np.nan
        if df_t.empty:
            return (
                pd.DataFrame({"metric":["Latency(ms)","Jitter(px)","MAE(px)","RMSE(px)"], "value":[lat_mean,np.nan,np.nan,np.nan]}),
                pd.DataFrame(columns=["point_id","jitter_px","mae_px","rmse_px","n"])
            )
        err_px = df_t["err_px"].values
        mae = float(np.mean(err_px))
        rmse = float(np.sqrt(np.mean(np.square(err_px))))
        jitter = float(np.std(err_px))
        df_global = pd.DataFrame({
            "metric": ["Latency(ms)","Jitter(px)","MAE(px)","RMSE(px)"],
            "value":  [lat_mean,      jitter,       mae,       rmse]
        })
        point_ids = df_t["point_id"].values.astype(int)
        unique_ids = np.unique(point_ids)
        jitter_px, mae_px, rmse_px, n = [], [], [], []
        @njit
        def calc_metrics(point_ids, err_px, unique_ids):
            jitter_px = []
            mae_px = []
            rmse_px = []
            n = []
            for pid in unique_ids:
                mask = point_ids == pid
                vals = err_px[mask]
                jitter_px.append(np.std(vals))
                mae_px.append(np.mean(vals))
                rmse_px.append(np.sqrt(np.mean(np.square(vals))))
                n.append(np.sum(mask))
            return jitter_px, mae_px, rmse_px, n

        jitter_px, mae_px, rmse_px, n = calc_metrics(point_ids, err_px, unique_ids)
        jitter_px = [float(x) for x in jitter_px]
        mae_px = [float(x) for x in mae_px]
        rmse_px = [float(x) for x in rmse_px]
        n = [int(x) for x in n]
        df_points = pd.DataFrame({
            "point_id": unique_ids,
            "jitter_px": jitter_px,
            "mae_px": mae_px,
            "rmse_px": rmse_px,
            "n": n
        }).sort_values("point_id").reset_index(drop=True)
        return df_global, df_points

    def save_csv(self, prefix="metrics"):
        if pd is None:
            return None, None
        df_global, df_points = self.summarize()
        ts = time.strftime("%Y%m%d-%H%M%S")
        p1 = f"{prefix}_global_{ts}.csv"
        p2 = f"{prefix}_points_{ts}.csv"
        os.makedirs(os.path.dirname(p1) or ".", exist_ok=True)
        df_global.to_csv(p1, index=False, encoding="utf-8-sig")
        df_points.to_csv(p2, index=False, encoding="utf-8-sig")
        return p1, p2

    def export_webcam_charts(self, *, export_dir="exports", deg_per_px: float | None = None) -> list[str]:
        if pd is None or plt is None:
            raise RuntimeError("pandas and matplotlib are required for PNG export")
        os.makedirs(export_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        df = self._df()
        if df is None or df.empty:
            raise RuntimeError("No metrics to export yet")
        df_global, df_points = self.summarize()
        saved = []

        # 1) Latency mean ± std
        import numpy as np
        lat_mean = float(np.nanmean(df["latency_ms"])) if "latency_ms" in df and len(df) else np.nan
        lat_std  = float(np.nanstd(df["latency_ms"])) if "latency_ms" in df and len(df) else np.nan
        fig = plt.figure(figsize=(10,6))
        plt.bar(["Latency_mean_ms"], [lat_mean])
        plt.errorbar([0], [lat_mean], yerr=[lat_std], fmt='k|', lw=2)
        plt.title("Latency (Webcam): mean ± std")
        plt.ylabel("Milliseconds")
        plt.tight_layout()
        p = f"{export_dir}/webcam_latency_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 2) Summary Jitter metrics
        if not df_points.empty:
            jitter_vals = df_points["jitter_px"].dropna().values
            j_mean = float(np.mean(jitter_vals)) if jitter_vals.size else np.nan
            j_median = float(np.median(jitter_vals)) if jitter_vals.size else np.nan
            j_std = float(np.std(jitter_vals)) if jitter_vals.size else np.nan
        else:
            j_mean = j_median = j_std = np.nan
        fig = plt.figure(figsize=(10,6))
        plt.bar(["Jitter_mean_px","Jitter_median_px","Jitter_std_px"], [j_mean, j_median, j_std])
        plt.title("Summary Jitter Metrics (Webcam)")
        plt.ylabel("Pixels")
        plt.tight_layout()
        p = f"{export_dir}/webcam_jitter_summary_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 3) Per-Point MAE
        fig = plt.figure(figsize=(10,6))
        if not df_points.empty:
            plt.bar(df_points["point_id"].astype(int).astype(str), df_points["mae_px"])
        plt.title("Per-Point Accuracy: MAE (Webcam)")
        plt.xlabel("Point Index (0–8)")
        plt.ylabel("MAE (pixels)")
        plt.tight_layout()
        p = f"{export_dir}/webcam_mae_per_point_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 4) Per-Point Jitter
        fig = plt.figure(figsize=(10,6))
        if not df_points.empty:
            plt.bar(df_points["point_id"].astype(int).astype(str), df_points["jitter_px"])
        plt.title("Per-Point Stability: Jitter Mean (Webcam)")
        plt.xlabel("Point Index (0–8)")
        plt.ylabel("Jitter Mean (pixels)")
        plt.tight_layout()
        p = f"{export_dir}/webcam_jitter_mean_per_point_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 5) Per-Point RMSE
        fig = plt.figure(figsize=(10,6))
        if not df_points.empty:
            plt.bar(df_points["point_id"].astype(int).astype(str), df_points["rmse_px"])
        plt.title("Per-Point Accuracy: RMSE (Webcam)")
        plt.xlabel("Point Index (0–8)")
        plt.ylabel("RMSE (pixels)")
        plt.tight_layout()
        p = f"{export_dir}/webcam_rmse_per_point_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 6) Summary Accuracy
        mae_px = float(df_global.loc[df_global["metric"]=="MAE(px)","value"].values[0])
        rmse_px = float(df_global.loc[df_global["metric"]=="RMSE(px)","value"].values[0])
        labels = ["MAE_px","RMSE_px"]; values = [mae_px, rmse_px]
        if (deg_per_px is not None) and (deg_per_px > 0):
            labels += ["MAE_deg","RMSE_deg"]
            values += [mae_px * deg_per_px, rmse_px * deg_per_px]
        fig = plt.figure(figsize=(10,6))
        plt.bar(labels, values)
        plt.title("Summary Accuracy Metrics (Webcam)")
        plt.ylabel("Value")
        plt.tight_layout()
        p = f"{export_dir}/webcam_accuracy_summary_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        return saved

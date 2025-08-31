
class CalibrationReport:
    def __init__(self, n_points=0, rmse_px=0.0, rmse_cv_px=0.0, uniformity=1.0, width=1, height=1):
        self.n_points = n_points
        self.rmse_px = rmse_px
        self.rmse_cv_px = rmse_cv_px
        self.uniformity = uniformity
        self.width = width
        self.height = height

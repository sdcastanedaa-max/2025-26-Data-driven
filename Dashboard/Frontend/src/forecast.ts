// frontend/src/forecast.ts

export interface ForecastPoint {
  time: Date;
  pvActual: number;
  pvForecast: number;
  windActual: number;
  windForecast: number;
}

export interface ErrorMetrics {
  mae: number;
  rmse: number;
  bias: number;
  mape: number | null;
}

function randn(mean = 0, std = 1): number {
  const u = 1 - Math.random();
  const v = 1 - Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return mean + z * std;
}

export function generateDummyForecast(
  nHours: number = 72,
): ForecastPoint[] {
  const now = new Date();
  now.setMinutes(0, 0, 0);

  const points: ForecastPoint[] = [];

  for (let i = 0; i < nHours; i++) {
    const t = i;
    const time = new Date(now.getTime() + i * 3600 * 1000);

    // PV: daytime bump
    const pvPattern = Math.max(
      0,
      Math.sin(((t % 24) - 6) / 24 * 2 * Math.PI),
    );
    const pvActual = 15000 * pvPattern + randn(0, 500);

    // Wind: slower oscillation
    const windPattern =
      0.6 +
      0.2 * Math.sin((t / 24) * 2 * Math.PI) +
      0.1 * Math.sin((t / 6) * 2 * Math.PI);
    const windActual =
      20000 * Math.max(0.1, windPattern) + randn(0, 800);

    const pvForecast = pvActual + randn(0, 600);
    const windForecast = windActual + randn(0, 900);

    points.push({
      time,
      pvActual,
      pvForecast,
      windActual,
      windForecast,
    });
  }

  return points;
}

export function computeErrorMetrics(
  yTrue: number[],
  yPred: number[],
): ErrorMetrics {
  const n = yTrue.length;
  if (n === 0 || n !== yPred.length) {
    return { mae: 0, rmse: 0, bias: 0, mape: null };
  }

  let mae = 0;
  let mse = 0;
  let bias = 0;
  let mapeSum = 0;
  let mapeCount = 0;

  for (let i = 0; i < n; i++) {
    const t = yTrue[i];
    const p = yPred[i];
    const diff = p - t;

    mae += Math.abs(diff);
    mse += diff * diff;
    bias += diff;

    if (t !== 0) {
      mapeSum += Math.abs(diff / t);
      mapeCount += 1;
    }
  }

  mae /= n;
  const rmse = Math.sqrt(mse / n);
  bias /= n;
  const mape = mapeCount > 0 ? (mapeSum / mapeCount) * 100 : null;

  return { mae, rmse, bias, mape };
}

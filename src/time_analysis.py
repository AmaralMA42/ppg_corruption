import numpy as np


def nanmean_silent(values, axis):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    count = np.sum(valid, axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )


def nansem_silent(values, axis):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    count = np.sum(valid, axis=axis)
    mean = nanmean_silent(values, axis=axis)
    expanded_mean = np.expand_dims(mean, axis=axis)
    centered = np.where(valid, values - expanded_mean, np.nan)
    variance = np.nansum(centered ** 2, axis=axis)
    variance = np.divide(
        variance,
        count - 1,
        out=np.full(mean.shape, np.nan, dtype=np.float64),
        where=count > 1,
    )
    return np.divide(
        np.sqrt(variance),
        np.sqrt(count),
        out=np.full(mean.shape, np.nan, dtype=np.float64),
        where=count > 1,
    )


def temporal_variance(series):
    return np.var(series, axis=-1)


def autocorrelation(series, normalize=True):
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.full(1, np.nan)

    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")[n - 1:]
    counts = np.arange(n, 0, -1)
    corr = corr / counts

    if normalize:
        if corr[0] <= 0:
            return np.full(n, np.nan)
        corr = corr / corr[0]

    return corr


def power_spectrum(series, dt=1.0):
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.full(1, np.nan), np.full(1, np.nan)

    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(np.fft.rfft(x)) ** 2 / n
    return freqs, power


def fixed_length_power_spectrum(series, n_fft, dt=1.0):
    x = np.asarray(series, dtype=np.float64)
    if x.size < 2 or n_fft < 2:
        return np.full(max(n_fft // 2 + 1, 1), np.nan)

    n = min(x.size, n_fft)
    x = x[:n] - np.mean(x[:n])
    padded = np.zeros(n_fft, dtype=np.float64)
    padded[:n] = x
    return np.abs(np.fft.rfft(padded)) ** 2 / n


def dominant_spectral_metrics(series, dt=1.0):
    freqs, power = power_spectrum(series, dt=dt)
    if freqs.size <= 1 or np.all(~np.isfinite(power[1:])):
        return np.nan, np.nan, np.nan, np.nan

    candidate_power = power[1:]
    total_power = np.nansum(candidate_power)
    if total_power <= 0 or np.nanmax(candidate_power) <= 0:
        return np.nan, np.nan, np.nan, np.nan

    idx = int(np.nanargmax(candidate_power)) + 1
    freq = freqs[idx]
    peak_power = power[idx]
    period = 1.0 / freq if freq > 0 else np.nan
    peak_ratio = peak_power / total_power
    return freq, peak_power, period, peak_ratio


def analyze_strategy_timeseries(estrat_t, start, absorbed_at=None, dt=1.0, min_points=4):
    n_strat, n_samples, total_passos = estrat_t.shape
    if absorbed_at is None:
        absorbed_at = np.full(n_samples, total_passos, dtype=np.int64)
    else:
        absorbed_at = np.asarray(absorbed_at, dtype=np.int64)

    max_lag = max(total_passos - start, 1)
    fft_freqs = np.fft.rfftfreq(max_lag, d=dt)
    variance_samples = np.full((n_strat, n_samples), np.nan)
    autocorr_samples = np.full((n_strat, n_samples, max_lag), np.nan)
    fft_power_samples = np.full((n_strat, n_samples, fft_freqs.size), np.nan)
    dominant_freq_samples = np.full((n_strat, n_samples), np.nan)
    dominant_power_samples = np.full((n_strat, n_samples), np.nan)
    dominant_period_samples = np.full((n_strat, n_samples), np.nan)
    peak_ratio_samples = np.full((n_strat, n_samples), np.nan)

    for sample in range(n_samples):
        stop = int(np.clip(absorbed_at[sample], 0, total_passos))
        if stop <= start:
            continue

        for strat in range(n_strat):
            series = estrat_t[strat, sample, start:stop]
            if series.size < min_points:
                continue

            variance_samples[strat, sample] = temporal_variance(series)
            corr = autocorrelation(series, normalize=True)
            autocorr_samples[strat, sample, :corr.size] = corr
            fft_power_samples[strat, sample, :] = fixed_length_power_spectrum(
                series,
                max_lag,
                dt=dt,
            )
            freq, power, period, peak_ratio = dominant_spectral_metrics(series, dt=dt)
            dominant_freq_samples[strat, sample] = freq
            dominant_power_samples[strat, sample] = power
            dominant_period_samples[strat, sample] = period
            peak_ratio_samples[strat, sample] = peak_ratio

    return {
        "variance_samples": variance_samples,
        "variance_mean": nanmean_silent(variance_samples, axis=1),
        "autocorr_samples": autocorr_samples,
        "autocorr_mean": nanmean_silent(autocorr_samples, axis=1),
        "fft_freqs": fft_freqs,
        "fft_power_samples": fft_power_samples,
        "fft_power_mean": nanmean_silent(fft_power_samples, axis=1),
        "fft_power_sem": nansem_silent(fft_power_samples, axis=1),
        "dominant_freq_samples": dominant_freq_samples,
        "dominant_freq_mean": nanmean_silent(dominant_freq_samples, axis=1),
        "dominant_power_samples": dominant_power_samples,
        "dominant_power_mean": nanmean_silent(dominant_power_samples, axis=1),
        "dominant_period_samples": dominant_period_samples,
        "dominant_period_mean": nanmean_silent(dominant_period_samples, axis=1),
        "peak_ratio_samples": peak_ratio_samples,
        "peak_ratio_mean": nanmean_silent(peak_ratio_samples, axis=1),
    }


def analyze_strategy_scalar_metrics(estrat_t, start, absorbed_at=None, dt=1.0, min_points=4):
    n_strat, n_samples, total_passos = estrat_t.shape
    if absorbed_at is None:
        absorbed_at = np.full(n_samples, total_passos, dtype=np.int64)
    else:
        absorbed_at = np.asarray(absorbed_at, dtype=np.int64)

    variance_samples = np.full((n_strat, n_samples), np.nan)
    dominant_freq_samples = np.full((n_strat, n_samples), np.nan)
    dominant_power_samples = np.full((n_strat, n_samples), np.nan)
    dominant_period_samples = np.full((n_strat, n_samples), np.nan)
    peak_ratio_samples = np.full((n_strat, n_samples), np.nan)

    for sample in range(n_samples):
        stop = int(np.clip(absorbed_at[sample], 0, total_passos))
        if stop <= start:
            continue

        for strat in range(n_strat):
            series = estrat_t[strat, sample, start:stop]
            if series.size < min_points:
                continue

            variance_samples[strat, sample] = temporal_variance(series)
            freq, power, period, peak_ratio = dominant_spectral_metrics(series, dt=dt)
            dominant_freq_samples[strat, sample] = freq
            dominant_power_samples[strat, sample] = power
            dominant_period_samples[strat, sample] = period
            peak_ratio_samples[strat, sample] = peak_ratio

    return {
        "variance_samples": variance_samples,
        "variance_mean": nanmean_silent(variance_samples, axis=1),
        "variance_sem": nansem_silent(variance_samples, axis=1),
        "dominant_freq_samples": dominant_freq_samples,
        "dominant_freq_mean": nanmean_silent(dominant_freq_samples, axis=1),
        "dominant_freq_sem": nansem_silent(dominant_freq_samples, axis=1),
        "dominant_power_samples": dominant_power_samples,
        "dominant_power_mean": nanmean_silent(dominant_power_samples, axis=1),
        "dominant_power_sem": nansem_silent(dominant_power_samples, axis=1),
        "dominant_period_samples": dominant_period_samples,
        "dominant_period_mean": nanmean_silent(dominant_period_samples, axis=1),
        "dominant_period_sem": nansem_silent(dominant_period_samples, axis=1),
        "peak_ratio_samples": peak_ratio_samples,
        "peak_ratio_mean": nanmean_silent(peak_ratio_samples, axis=1),
        "peak_ratio_sem": nansem_silent(peak_ratio_samples, axis=1),
    }

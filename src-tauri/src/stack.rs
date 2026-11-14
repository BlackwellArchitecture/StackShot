//! High-performance image stacking engine for StackShot.
//!
//! All pixel-level statistical stacking happens here in Rust using Rayon
//! parallel row-band processing (`par_chunks_mut`), single-pass online accumulators
//! (Welford/Pébay moments for Mean, Variance, StdDev, Skewness, Kurtosis),
//! and `select_nth_unstable` for $O(N)$ Median selection without full sorting.

use image::{imageops::FilterType, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Supported Photoshop-style and scientific image stack modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StackMode {
    Median,
    Mean,
    Maximum,
    Minimum,
    Range,
    StdDev,
    Summation,
    Variance,
    Skewness,
    Kurtosis,
    Entropy,
}

impl StackMode {
    /// Returns true if this mode produces low-amplitude statistical values
    /// that benefit from automatic 0..=255 contrast normalization.
    pub fn is_low_output_mode(self) -> bool {
        matches!(
            self,
            StackMode::Range
                | StackMode::StdDev
                | StackMode::Variance
                | StackMode::Skewness
                | StackMode::Kurtosis
                | StackMode::Entropy
        )
    }

    /// Human-readable label for UI display.
    pub fn label(self) -> &'static str {
        match self {
            StackMode::Median => "Median",
            StackMode::Mean => "Mean (Average)",
            StackMode::Maximum => "Maximum",
            StackMode::Minimum => "Minimum",
            StackMode::Range => "Range",
            StackMode::StdDev => "Standard Deviation",
            StackMode::Summation => "Summation",
            StackMode::Variance => "Variance",
            StackMode::Skewness => "Skewness",
            StackMode::Kurtosis => "Kurtosis",
            StackMode::Entropy => "Entropy",
        }
    }

    /// One-line description explaining what the mode does and when to use it.
    pub fn description(self) -> &'static str {
        match self {
            StackMode::Median => {
                "Picks the middle value per pixel — ideal for removing moving objects, tourists, traffic, or transient noise."
            }
            StackMode::Mean => {
                "Averages pixel values across frames — simulates long exposures, smooths water/clouds, and reduces sensor noise."
            }
            StackMode::Maximum => {
                "Keeps the brightest pixel value — great for star trails, fireworks, light painting, and lightning composites."
            }
            StackMode::Minimum => {
                "Keeps the darkest pixel value — removes bright flashes, specular glare, or light-colored moving objects."
            }
            StackMode::Range => {
                "Computes Maximum minus Minimum — highlights total motion amplitude and checks for tripod shift."
            }
            StackMode::StdDev => {
                "Measures variation from the mean — used in scientific imaging to isolate movement and temporal change."
            }
            StackMode::Summation => {
                "Adds pixel intensities (clamped to 255) — boosts faint signals in astrophotography and low-light captures."
            }
            StackMode::Variance => {
                "Squared deviation from the mean — accentuates high-turbulence regions and strong temporal fluctuations."
            }
            StackMode::Skewness => {
                "Measures asymmetry of pixel distribution over time — detects brief bright or dark directional anomalies."
            }
            StackMode::Kurtosis => {
                "Measures outlier extremity (tailedness) across frames — isolates rare, sudden spikes against steady backgrounds."
            }
            StackMode::Entropy => {
                "Computes Shannon information entropy across frames — maps temporal randomness and visual complexity."
            }
        }
    }
}

/// Strategy for determining target output canvas dimensions when inputs differ in size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SizeMatchMode {
    /// Match dimensions to the first image in the stack (default).
    First,
    /// Match dimensions to the largest width and height present in the stack.
    Largest,
}

impl Default for SizeMatchMode {
    fn default() -> Self {
        SizeMatchMode::First
    }
}

/// How an image with a different aspect ratio is fitted into the target canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitMode {
    /// Scale to fill target canvas and crop overflow from the center (default).
    CenterCrop,
    /// Scale to fit inside target canvas and pad remaining area with transparent/black bars.
    Letterbox,
}

impl Default for FitMode {
    fn default() -> Self {
        FitMode::CenterCrop
    }
}

/// Configuration parameters for a stack operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackOptions {
    pub mode: StackMode,
    pub auto_normalize: bool,
    pub size_match: SizeMatchMode,
    pub fit_mode: FitMode,
}

impl Default for StackOptions {
    fn default() -> Self {
        Self {
            mode: StackMode::Median,
            auto_normalize: true,
            size_match: SizeMatchMode::First,
            fit_mode: FitMode::CenterCrop,
        }
    }
}

/// Lightweight view of a raw RGBA8 image buffer in memory.
#[derive(Debug, Clone)]
pub struct RawImage {
    pub width: u32,
    pub height: u32,
    /// Contiguous RGBA8 bytes (`width * height * 4`).
    pub data: Vec<u8>,
    /// True if any pixel in `data` has alpha < 255.
    pub has_alpha: bool,
}

impl RawImage {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize) * 4,
            "RawImage buffer size mismatch"
        );
        let has_alpha = data.chunks_exact(4).any(|px| px[3] < 255);
        Self {
            width,
            height,
            data,
            has_alpha,
        }
    }

    pub fn with_known_alpha(width: u32, height: u32, data: Vec<u8>, has_alpha: bool) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize) * 4,
            "RawImage buffer size mismatch"
        );
        Self {
            width,
            height,
            data,
            has_alpha,
        }
    }
}

/// Output of a completed stack operation.
#[derive(Debug, Clone)]
pub struct StackResult {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub render_ms: f64,
}

/// Error types that can occur during stacking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StackError {
    NotEnoughImages,
    InvalidDimensions,
    Cancelled,
}

impl std::fmt::Display for StackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StackError::NotEnoughImages => {
                write!(f, "At least 2 images are required to compute a stack.")
            }
            StackError::InvalidDimensions => write!(f, "Image dimensions must be non-zero."),
            StackError::Cancelled => write!(f, "Stack computation was cancelled."),
        }
    }
}

impl std::error::Error for StackError {}

/// Determines target `(width, height)` based on input images and `SizeMatchMode`.
pub fn resolve_target_dimensions(images: &[RawImage], size_match: SizeMatchMode) -> (u32, u32) {
    if images.is_empty() {
        return (0, 0);
    }
    match size_match {
        SizeMatchMode::First => (images[0].width, images[0].height),
        SizeMatchMode::Largest => {
            let max_w = images.iter().map(|img| img.width).max().unwrap_or(0);
            let max_h = images.iter().map(|img| img.height).max().unwrap_or(0);
            (max_w, max_h)
        }
    }
}

/// Resizes/fits a single `RawImage` to `(target_w, target_h)` using `FitMode`.
/// If the image already matches `(target_w, target_h)`, borrows the original buffer with zero copy.
pub fn normalize_image_size<'a>(
    img: &'a RawImage,
    target_w: u32,
    target_h: u32,
    fit_mode: FitMode,
) -> Cow<'a, [u8]> {
    if img.width == target_w && img.height == target_h {
        return Cow::Borrowed(&img.data);
    }

    let src = RgbaImage::from_raw(img.width, img.height, img.data.clone())
        .expect("Valid RGBA8 buffer for source image");

    let src_w = img.width as f64;
    let src_h = img.height as f64;
    let dst_w = target_w as f64;
    let dst_h = target_h as f64;

    match fit_mode {
        FitMode::CenterCrop => {
            // Scale so both dimensions cover at least (target_w, target_h)
            let scale = (dst_w / src_w).max(dst_h / src_h);
            let scaled_w = ((src_w * scale).round() as u32).max(target_w);
            let scaled_h = ((src_h * scale).round() as u32).max(target_h);

            let resized = image::imageops::resize(&src, scaled_w, scaled_h, FilterType::Triangle);
            if scaled_w == target_w && scaled_h == target_h {
                return Cow::Owned(resized.into_raw());
            }

            let offset_x = (scaled_w - target_w) / 2;
            let offset_y = (scaled_h - target_h) / 2;
            let cropped =
                image::imageops::crop_imm(&resized, offset_x, offset_y, target_w, target_h)
                    .to_image();
            Cow::Owned(cropped.into_raw())
        }
        FitMode::Letterbox => {
            // Scale so both dimensions fit within (target_w, target_h)
            let scale = (dst_w / src_w).min(dst_h / src_h);
            let scaled_w = ((src_w * scale).round() as u32).clamp(1, target_w);
            let scaled_h = ((src_h * scale).round() as u32).clamp(1, target_h);

            let resized = image::imageops::resize(&src, scaled_w, scaled_h, FilterType::Triangle);
            let mut canvas = RgbaImage::from_pixel(
                target_w,
                target_h,
                image::Rgba([0, 0, 0, if img.has_alpha { 0 } else { 255 }]),
            );
            let offset_x = ((target_w - scaled_w) / 2) as i64;
            let offset_y = ((target_h - scaled_h) / 2) as i64;
            image::imageops::overlay(&mut canvas, &resized, offset_x, offset_y);
            Cow::Owned(canvas.into_raw())
        }
    }
}

/// Computes the median of `buf` in $O(N)$ average time using `select_nth_unstable` (no full sort).
#[inline]
pub fn fast_median_u8(buf: &mut [u8]) -> u8 {
    let n = buf.len();
    debug_assert!(n > 0);
    if n == 1 {
        return buf[0];
    }
    let mid = n / 2;
    let (left, &mut upper, _) = buf.select_nth_unstable(mid);
    if n % 2 == 1 {
        upper
    } else {
        // `left` has all elements with index < mid, all guaranteed <= `upper`.
        let lower = left.iter().copied().max().unwrap_or(upper);
        ((lower as u16 + upper as u16) / 2) as u8
    }
}

/// Core parallel stacking function.
///
/// - Processes horizontal row-bands in parallel via Rayon `par_chunks_mut`.
/// - Single-pass accumulators for Mean, Min, Max, Range, Summation, StdDev, Variance, Skewness, Kurtosis.
/// - `select_nth_unstable` for Median without sorting.
/// - Checks `cancel_flag` regularly and invokes `on_progress(percentage)` as bands complete.
pub fn stack_images<F>(
    images: &[RawImage],
    options: &StackOptions,
    cancel_flag: Option<&Arc<AtomicBool>>,
    on_progress: F,
) -> Result<StackResult, StackError>
where
    F: Fn(u8) + Send + Sync,
{
    let start_time = Instant::now();

    if images.len() < 2 {
        return Err(StackError::NotEnoughImages);
    }

    let (target_w, target_h) = resolve_target_dimensions(images, options.size_match);
    if target_w == 0 || target_h == 0 {
        return Err(StackError::InvalidDimensions);
    }

    let is_cancelled = || -> bool {
        cancel_flag
            .map(|flag| flag.load(Ordering::Relaxed))
            .unwrap_or(false)
    };

    if is_cancelled() {
        return Err(StackError::Cancelled);
    }

    on_progress(2);

    // Step 1: Prepare aligned slices (zero-copy Cow::Borrowed when sizes already match).
    let aligned_buffers: Result<Vec<Cow<'_, [u8]>>, StackError> = images
        .par_iter()
        .map(|img| {
            if is_cancelled() {
                return Err(StackError::Cancelled);
            }
            Ok(normalize_image_size(img, target_w, target_h, options.fit_mode))
        })
        .collect();

    let aligned_buffers = aligned_buffers?;
    let slices: Vec<&[u8]> = aligned_buffers.iter().map(|c| c.as_ref()).collect();
    let num_images = slices.len();
    let all_have_alpha = images.iter().all(|img| img.has_alpha);

    if is_cancelled() {
        return Err(StackError::Cancelled);
    }

    on_progress(10);

    // Step 2: Preallocate single contiguous output RGBA8 buffer.
    let total_bytes = (target_w as usize) * (target_h as usize) * 4;
    let mut output = vec![0u8; total_bytes];

    // Process in horizontal bands of rows so CPU cores stay saturated with minimal overhead.
    let row_bytes = (target_w as usize) * 4;
    let rows_per_band = (64usize).clamp(1, (target_h as usize / 16).max(1));
    let band_bytes = row_bytes * rows_per_band;
    let total_bands = (total_bytes + band_bytes - 1) / band_bytes;
    let completed_bands = AtomicUsize::new(0);
    let last_reported_pct = AtomicUsize::new(10);

    let mode = options.mode;

    // Step 3: Parallel row-band processing.
    let band_result: Result<(), StackError> = output
        .par_chunks_mut(band_bytes)
        .enumerate()
        .try_for_each(|(band_idx, out_band)| {
            if is_cancelled() {
                return Err(StackError::Cancelled);
            }

            let byte_offset_start = band_idx * band_bytes;

            // Reusable thread-local pixel sample buffer for Median/Entropy modes
            let mut sample_buf = vec![0u8; num_images];

            for (px_idx, out_px) in out_band.chunks_exact_mut(4).enumerate() {
                let global_px_byte = byte_offset_start + px_idx * 4;

                // Compute R, G, B independently
                for c in 0..3 {
                    let ch_offset = global_px_byte + c;
                    out_px[c] = compute_channel_stat(mode, &slices, ch_offset, &mut sample_buf);
                }

                // Alpha handling: opaque (255) unless ALL inputs have alpha
                if all_have_alpha {
                    let alpha_offset = global_px_byte + 3;
                    out_px[3] = match mode {
                        StackMode::Median
                        | StackMode::Mean
                        | StackMode::Maximum
                        | StackMode::Minimum
                        | StackMode::Summation => {
                            compute_channel_stat(mode, &slices, alpha_offset, &mut sample_buf)
                        }
                        // For statistical variation modes, preserve mean alpha so differences stay visible
                        _ => compute_channel_stat(
                            StackMode::Mean,
                            &slices,
                            alpha_offset,
                            &mut sample_buf,
                        ),
                    };
                } else {
                    out_px[3] = 255;
                }
            }

            let done = completed_bands.fetch_add(1, Ordering::Relaxed) + 1;
            let pct = 10 + (done * 82) / total_bands.max(1);
            let prev = last_reported_pct.load(Ordering::Relaxed);
            if pct > prev && last_reported_pct.compare_exchange(prev, pct, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                on_progress(pct as u8);
            }

            Ok(())
        });

    band_result?;

    if is_cancelled() {
        return Err(StackError::Cancelled);
    }

    // Step 4: Optional Auto-Normalize for low-output statistical modes
    if options.auto_normalize && mode.is_low_output_mode() {
        auto_normalize_rgb_in_place(&mut output);
    }

    on_progress(100);

    let render_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok(StackResult {
        width: target_w,
        height: target_h,
        rgba: output,
        render_ms,
    })
}

/// Computes the requested statistical mode for a single pixel channel across all input slices.
#[inline(always)]
fn compute_channel_stat(
    mode: StackMode,
    slices: &[&[u8]],
    ch_offset: usize,
    sample_buf: &mut [u8],
) -> u8 {
    let n = slices.len();
    match mode {
        StackMode::Mean => {
            // Single-pass integer accumulator
            let mut sum: u64 = 0;
            for s in slices {
                sum += unsafe { *s.get_unchecked(ch_offset) } as u64;
            }
            ((sum + (n as u64 / 2)) / (n as u64)) as u8
        }
        StackMode::Maximum => {
            let mut max_v = 0u8;
            for s in slices {
                let v = unsafe { *s.get_unchecked(ch_offset) };
                if v > max_v {
                    max_v = v;
                }
            }
            max_v
        }
        StackMode::Minimum => {
            let mut min_v = 255u8;
            for s in slices {
                let v = unsafe { *s.get_unchecked(ch_offset) };
                if v < min_v {
                    min_v = v;
                }
            }
            min_v
        }
        StackMode::Range => {
            let mut min_v = 255u8;
            let mut max_v = 0u8;
            for s in slices {
                let v = unsafe { *s.get_unchecked(ch_offset) };
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            max_v - min_v
        }
        StackMode::Summation => {
            let mut sum: u32 = 0;
            for s in slices {
                sum += unsafe { *s.get_unchecked(ch_offset) } as u32;
            }
            sum.min(255) as u8
        }
        StackMode::Median => {
            for (i, s) in slices.iter().enumerate() {
                sample_buf[i] = unsafe { *s.get_unchecked(ch_offset) };
            }
            fast_median_u8(sample_buf)
        }
        StackMode::StdDev => {
            // Single-pass Welford's algorithm for population standard deviation
            let mut count = 0.0_f64;
            let mut mean = 0.0_f64;
            let mut m2 = 0.0_f64;
            for s in slices {
                let x = unsafe { *s.get_unchecked(ch_offset) } as f64;
                count += 1.0;
                let delta = x - mean;
                mean += delta / count;
                let delta2 = x - mean;
                m2 += delta * delta2;
            }
            let variance = m2 / count;
            variance.max(0.0).sqrt().round().clamp(0.0, 255.0) as u8
        }
        StackMode::Variance => {
            // Single-pass Welford's algorithm for population variance
            let mut count = 0.0_f64;
            let mut mean = 0.0_f64;
            let mut m2 = 0.0_f64;
            for s in slices {
                let x = unsafe { *s.get_unchecked(ch_offset) } as f64;
                count += 1.0;
                let delta = x - mean;
                mean += delta / count;
                let delta2 = x - mean;
                m2 += delta * delta2;
            }
            let variance = (m2 / count).max(0.0);
            if variance <= 1e-9 {
                0
            } else {
                // Max theoretical variance on [0, 255] is 127.5^2 = 16256.25.
                // Divide by 63.75 to map [0, 16256.25] -> [0, 255], clamping non-zero values to >= 1.
                (variance / 63.75).round().clamp(1.0, 255.0) as u8
            }
        }
        StackMode::Skewness => {
            // Single-pass online 3rd central moment
            let mut count = 0.0_f64;
            let mut mean = 0.0_f64;
            let mut m2 = 0.0_f64;
            let mut m3 = 0.0_f64;
            for s in slices {
                let x = unsafe { *s.get_unchecked(ch_offset) } as f64;
                let n1 = count;
                count += 1.0;
                let delta = x - mean;
                let delta_n = delta / count;
                let term1 = delta * delta_n * n1;
                mean += delta_n;
                m3 += term1 * delta_n * (count - 2.0) - 3.0 * delta_n * m2;
                m2 += term1;
            }
            if m2 <= 1e-9 {
                0
            } else {
                let skew = (count.sqrt() * m3) / m2.powf(1.5);
                // Map magnitude of skewness into 0..=255
                ((skew.abs() / 3.0) * 255.0).round().clamp(0.0, 255.0) as u8
            }
        }
        StackMode::Kurtosis => {
            // Single-pass online 4th central moment
            let mut count = 0.0_f64;
            let mut mean = 0.0_f64;
            let mut m2 = 0.0_f64;
            let mut m3 = 0.0_f64;
            let mut m4 = 0.0_f64;
            for s in slices {
                let x = unsafe { *s.get_unchecked(ch_offset) } as f64;
                let n1 = count;
                count += 1.0;
                let delta = x - mean;
                let delta_n = delta / count;
                let delta_n2 = delta_n * delta_n;
                let term1 = delta * delta_n * n1;
                mean += delta_n;
                m4 += term1 * delta_n2 * (count * count - 3.0 * count + 3.0)
                    + 6.0 * delta_n2 * m2
                    - 4.0 * delta_n * m3;
                m3 += term1 * delta_n * (count - 2.0) - 3.0 * delta_n * m2;
                m2 += term1;
            }
            if m2 <= 1e-9 {
                0
            } else {
                // Excess kurtosis = (N * M4 / M2^2) - 3.0; shifted & scaled for display
                let excess_kurt = ((count * m4) / (m2 * m2) - 3.0).max(0.0);
                ((excess_kurt / 6.0) * 255.0).round().clamp(0.0, 255.0) as u8
            }
        }
        StackMode::Entropy => {
            // Shannon entropy across stack samples quantized into 32 bins (8 levels per bin)
            let mut bins = [0u16; 32];
            for s in slices {
                let v = unsafe { *s.get_unchecked(ch_offset) };
                bins[(v >> 3) as usize] += 1;
            }
            let inv_n = 1.0 / (n as f64);
            let mut entropy = 0.0_f64;
            for &b in &bins {
                if b > 0 {
                    let p = (b as f64) * inv_n;
                    entropy -= p * p.log2();
                }
            }
            let max_entropy = (n.min(32) as f64).log2().max(1.0);
            ((entropy / max_entropy) * 255.0).round().clamp(0.0, 255.0) as u8
        }
    }
}

/// Stretches the RGB channels of `rgba` in-place so `[min_val, max_val]` spans `0..=255`.
/// If all pixels have the exact same value (`min_val == max_val`), leaves the buffer untouched
/// so uniform images (e.g. StdDev = 0 on identical images, or Range = 255 on black/white) stay exact.
pub fn auto_normalize_rgb_in_place(rgba: &mut [u8]) {
    if rgba.is_empty() {
        return;
    }

    let (min_v, max_v) = rgba
        .par_chunks(16384)
        .map(|chunk| {
            let mut local_min = 255u8;
            let mut local_max = 0u8;
            for px in chunk.chunks_exact(4) {
                for &c in &px[0..3] {
                    if c < local_min {
                        local_min = c;
                    }
                    if c > local_max {
                        local_max = c;
                    }
                }
            }
            (local_min, local_max)
        })
        .reduce(
            || (255u8, 0u8),
            |(min_a, max_a), (min_b, max_b)| (min_a.min(min_b), max_a.max(max_b)),
        );

    if max_v <= min_v {
        return;
    }

    if min_v == 0 && max_v == 255 {
        return;
    }

    let range = (max_v - min_v) as f32;
    let scale = 255.0 / range;

    // Precompute 256-entry LUT for instant parallel remapping
    let mut lut = [0u8; 256];
    for i in min_v..=max_v {
        lut[i as usize] = (((i - min_v) as f32) * scale).round().clamp(0.0, 255.0) as u8;
    }

    rgba.par_chunks_mut(16384).for_each(|chunk| {
        for px in chunk.chunks_exact_mut(4) {
            px[0] = lut[px[0] as usize];
            px[1] = lut[px[1] as usize];
            px[2] = lut[px[2] as usize];
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid_image(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> RawImage {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            data.extend_from_slice(&[r, g, b, a]);
        }
        RawImage::new(w, h, data)
    }

    #[test]
    fn test_mean_black_and_white_is_mid_gray() {
        let black = make_solid_image(4, 4, 0, 0, 0, 255);
        let white = make_solid_image(4, 4, 255, 255, 255, 255);
        let opts = StackOptions {
            mode: StackMode::Mean,
            ..Default::default()
        };

        let res = stack_images(&[black, white], &opts, None, |_| {}).unwrap();
        for px in res.rgba.chunks_exact(4) {
            assert!(
                px[0] == 127 || px[0] == 128,
                "Expected mid-gray (127 or 128), got {}",
                px[0]
            );
            assert!(px[1] == 127 || px[1] == 128);
            assert!(px[2] == 127 || px[2] == 128);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn test_maximum_black_and_white_is_white() {
        let black = make_solid_image(4, 4, 0, 0, 0, 255);
        let white = make_solid_image(4, 4, 255, 255, 255, 255);
        let opts = StackOptions {
            mode: StackMode::Maximum,
            ..Default::default()
        };

        let res = stack_images(&[black, white], &opts, None, |_| {}).unwrap();
        for px in res.rgba.chunks_exact(4) {
            assert_eq!(px, &[255, 255, 255, 255]);
        }
    }

    #[test]
    fn test_minimum_black_and_white_is_black() {
        let black = make_solid_image(4, 4, 0, 0, 0, 255);
        let white = make_solid_image(4, 4, 255, 255, 255, 255);
        let opts = StackOptions {
            mode: StackMode::Minimum,
            ..Default::default()
        };

        let res = stack_images(&[black, white], &opts, None, |_| {}).unwrap();
        for px in res.rgba.chunks_exact(4) {
            assert_eq!(px, &[0, 0, 0, 255]);
        }
    }

    #[test]
    fn test_range_black_and_white_is_white() {
        let black = make_solid_image(4, 4, 0, 0, 0, 255);
        let white = make_solid_image(4, 4, 255, 255, 255, 255);
        let opts = StackOptions {
            mode: StackMode::Range,
            auto_normalize: true,
            ..Default::default()
        };

        let res = stack_images(&[black, white], &opts, None, |_| {}).unwrap();
        for px in res.rgba.chunks_exact(4) {
            assert_eq!(px, &[255, 255, 255, 255]);
        }
    }

    #[test]
    fn test_median_of_three_removes_transient_object() {
        // Background is dark gray (40, 40, 40) across all 3 frames.
        // Frame 2 has a bright white "transient object" (255, 255, 255) in the center pixels.
        let img1 = make_solid_image(4, 4, 40, 40, 40, 255);
        let mut img2 = make_solid_image(4, 4, 40, 40, 40, 255);
        let img3 = make_solid_image(4, 4, 40, 40, 40, 255);

        // Paint transient object at pixel (1, 1) and (2, 2) in img2
        for idx in [5usize, 10usize] {
            let base = idx * 4;
            img2.data[base] = 255;
            img2.data[base + 1] = 255;
            img2.data[base + 2] = 255;
        }

        let opts = StackOptions {
            mode: StackMode::Median,
            ..Default::default()
        };

        let res = stack_images(&[img1, img2, img3], &opts, None, |_| {}).unwrap();
        for px in res.rgba.chunks_exact(4) {
            assert_eq!(
                px,
                &[40, 40, 40, 255],
                "Median of 3 images should completely remove the object present in only 1 image"
            );
        }
    }

    #[test]
    fn test_std_dev_of_identical_images_is_zero() {
        let img1 = make_solid_image(4, 4, 123, 80, 200, 255);
        let img2 = make_solid_image(4, 4, 123, 80, 200, 255);
        let img3 = make_solid_image(4, 4, 123, 80, 200, 255);

        let opts = StackOptions {
            mode: StackMode::StdDev,
            auto_normalize: true,
            ..Default::default()
        };

        let res = stack_images(&[img1, img2, img3], &opts, None, |_| {}).unwrap();
        for px in res.rgba.chunks_exact(4) {
            assert_eq!(px, &[0, 0, 0, 255]);
        }
    }

    #[test]
    fn test_mismatched_dimensions_and_alpha_rules() {
        let img1 = make_solid_image(4, 4, 100, 100, 100, 128);
        let img2 = make_solid_image(8, 8, 200, 200, 200, 255);

        // First image match (4x4), only img1 has alpha < 255 so output alpha must be opaque (255)
        let opts = StackOptions {
            mode: StackMode::Mean,
            size_match: SizeMatchMode::First,
            fit_mode: FitMode::CenterCrop,
            auto_normalize: false,
        };
        let res = stack_images(&[img1.clone(), img2.clone()], &opts, None, |_| {}).unwrap();
        assert_eq!((res.width, res.height), (4, 4));
        assert_eq!(res.rgba[3], 255);

        // Largest image match (8x8)
        let opts_largest = StackOptions {
            size_match: SizeMatchMode::Largest,
            fit_mode: FitMode::Letterbox,
            ..opts
        };
        let res_largest = stack_images(&[img1, img2], &opts_largest, None, |_| {}).unwrap();
        assert_eq!((res_largest.width, res_largest.height), (8, 8));
    }
}

//! StackShot Tauri backend library.
//! Manages in-memory RGBA8 source images, clipboard I/O, background Rayon stacking,
//! progress reporting, cancellation, and multi-format export (PNG, JPEG, WebP).

pub mod config;
pub mod stack;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use config::AppConfig;
use image::{
    codecs::{jpeg::JpegEncoder, png::PngEncoder, webp::WebPEncoder},
    imageops::FilterType,
    ColorType, ImageEncoder, RgbaImage,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use stack::{RawImage, StackError, StackMode, StackOptions, StackResult};
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Cursor};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

/// Maximum dimension (width or height) for thumbnail strip previews sent to the frontend.
const THUMBNAIL_MAX_DIM: u32 = 160;

/// Maximum dimension for the main viewport preview image (full-res stays in Rust memory).
const PREVIEW_MAX_DIM: u32 = 2048;

/// Warn user if raw source + working memory exceeds ~1.6 GB.
const MEMORY_WARNING_THRESHOLD_BYTES: u64 = 1_600 * 1024 * 1024;

/// A decoded source image held in Rust memory.
pub struct StoredImage {
    pub id: u64,
    pub name: String,
    pub raw: RawImage,
    pub thumbnail_data_url: String,
}

/// Metadata sent to the frontend for each image in the stack (never contains full-res bytes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMeta {
    pub id: u64,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
    pub byte_size: u64,
    pub thumbnail_data_url: String,
}

impl StoredImage {
    pub fn to_meta(&self) -> ImageMeta {
        ImageMeta {
            id: self.id,
            name: self.name.clone(),
            width: self.raw.width,
            height: self.raw.height,
            has_alpha: self.raw.has_alpha,
            byte_size: self.raw.data.len() as u64,
            thumbnail_data_url: self.thumbnail_data_url.clone(),
        }
    }
}

/// Summary of the current image stack and memory consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackStatusPayload {
    pub images: Vec<ImageMeta>,
    pub total_memory_bytes: u64,
    pub estimated_peak_bytes: u64,
    pub memory_warning: bool,
}

/// Static metadata about each stack mode for the frontend UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeInfo {
    pub id: StackMode,
    pub label: String,
    pub description: String,
    pub is_low_output: bool,
}

/// Initial state returned to the frontend on startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialStatePayload {
    pub config: AppConfig,
    pub modes: Vec<ModeInfo>,
    pub status: StackStatusPayload,
}

/// Progress event payload emitted to the frontend during stacking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressPayload {
    pub percent: u8,
    pub stage: String,
}

/// Result returned to the frontend when a stack operation completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackPreviewResponse {
    pub width: u32,
    pub height: u32,
    pub preview_width: u32,
    pub preview_height: u32,
    pub render_ms: f64,
    pub preview_data_url: String,
    pub mode_label: String,
}

/// Result returned when saving an image to disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveResultResponse {
    pub path: String,
    pub bytes_written: u64,
}

/// Shared application state managed by Tauri.
pub struct AppState {
    pub images: Mutex<Vec<StoredImage>>,
    pub last_result: Mutex<Option<StackResult>>,
    pub cancel_flag: Arc<AtomicBool>,
    pub next_id: AtomicU64,
    pub config: Mutex<AppConfig>,
    pub config_dir: PathBuf,
}

impl AppState {
    pub fn new(config_dir: PathBuf) -> Self {
        let config = AppConfig::load(&config_dir);
        Self {
            images: Mutex::new(Vec::new()),
            last_result: Mutex::new(None),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            next_id: AtomicU64::new(1),
            config: Mutex::new(config),
            config_dir,
        }
    }

    pub fn build_status(&self, images: &[StoredImage]) -> StackStatusPayload {
        let metas: Vec<ImageMeta> = images.iter().map(|img| img.to_meta()).collect();
        let total_memory_bytes: u64 = metas.iter().map(|m| m.byte_size).sum();
        let max_single_bytes: u64 = metas.iter().map(|m| m.byte_size).max().unwrap_or(0);
        // Peak memory includes source buffers + output buffer + any resized intermediate buffers
        let estimated_peak_bytes = total_memory_bytes + max_single_bytes * 2;
        let memory_warning = estimated_peak_bytes > MEMORY_WARNING_THRESHOLD_BYTES;

        StackStatusPayload {
            images: metas,
            total_memory_bytes,
            estimated_peak_bytes,
            memory_warning,
        }
    }
}

/// Encodes an RGBA8 image into a PNG data URL, downscaling if larger than `max_dim`.
fn encode_png_data_url(
    width: u32,
    height: u32,
    rgba: &[u8],
    max_dim: u32,
) -> Result<(String, u32, u32), String> {
    let img = RgbaImage::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| "Invalid RGBA buffer dimensions".to_string())?;

    let (out_img, out_w, out_h) = if width > max_dim || height > max_dim {
        let scale = (max_dim as f64 / width as f64).min(max_dim as f64 / height as f64);
        let new_w = ((width as f64 * scale).round() as u32).max(1);
        let new_h = ((height as f64 * scale).round() as u32).max(1);
        let resized = image::imageops::resize(&img, new_w, new_h, FilterType::Triangle);
        (resized, new_w, new_h)
    } else {
        (img, width, height)
    };

    let mut png_bytes = Vec::new();
    {
        let cursor = Cursor::new(&mut png_bytes);
        let encoder = PngEncoder::new(cursor);
        encoder
            .write_image(out_img.as_raw(), out_w, out_h, ColorType::Rgba8.into())
            .map_err(|e| format!("Failed to encode PNG preview: {e}"))?;
    }

    let b64 = BASE64.encode(&png_bytes);
    Ok((format!("data:image/png;base64,{b64}"), out_w, out_h))
}

/// Creates a `StoredImage` from raw RGBA8 data.
fn create_stored_image(
    id: u64,
    name: String,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
) -> Result<StoredImage, String> {
    let (thumbnail_data_url, _, _) = encode_png_data_url(width, height, &rgba, THUMBNAIL_MAX_DIM)?;
    let raw = RawImage::new(width, height, rgba);
    Ok(StoredImage {
        id,
        name,
        raw,
        thumbnail_data_url,
    })
}

#[tauri::command]
fn get_initial_state(state: State<'_, AppState>) -> Result<InitialStatePayload, String> {
    let config = state
        .config
        .lock()
        .map_err(|_| "Failed to lock config".to_string())?
        .clone();
    let images = state
        .images
        .lock()
        .map_err(|_| "Failed to lock image store".to_string())?;
    let status = state.build_status(&images);

    let all_modes = [
        StackMode::Median,
        StackMode::Mean,
        StackMode::Maximum,
        StackMode::Minimum,
        StackMode::Range,
        StackMode::StdDev,
        StackMode::Summation,
        StackMode::Variance,
        StackMode::Skewness,
        StackMode::Kurtosis,
        StackMode::Entropy,
    ];

    let modes = all_modes
        .iter()
        .map(|&m| ModeInfo {
            id: m,
            label: m.label().to_string(),
            description: m.description().to_string(),
            is_low_output: m.is_low_output_mode(),
        })
        .collect();

    Ok(InitialStatePayload {
        config,
        modes,
        status,
    })
}

#[tauri::command]
fn paste_clipboard_image(state: State<'_, AppState>) -> Result<StackStatusPayload, String> {
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| format!("Could not access clipboard: {e}"))?;

    let clip_img = clipboard.get_image().map_err(|_| {
        "No image found on the clipboard. Take a screenshot (Win+Shift+S / Cmd+Shift+4) or copy an image first.".to_string()
    })?;

    let width = clip_img.width as u32;
    let height = clip_img.height as u32;
    if width == 0 || height == 0 {
        return Err("Clipboard image has zero dimensions.".to_string());
    }

    let rgba = clip_img.bytes.into_owned();
    let id = state.next_id.fetch_add(1, Ordering::Relaxed);
    let name = format!("Clipboard #{id}");

    let stored = create_stored_image(id, name, width, height, rgba)?;

    let mut images = state
        .images
        .lock()
        .map_err(|_| "Failed to lock image store".to_string())?;
    images.push(stored);
    Ok(state.build_status(&images))
}

#[tauri::command]
async fn add_image_files(
    paths: Vec<String>,
    state: State<'_, AppState>,
) -> Result<StackStatusPayload, String> {
    if paths.is_empty() {
        let images = state
            .images
            .lock()
            .map_err(|_| "Failed to lock image store".to_string())?;
        return Ok(state.build_status(&images));
    }

    // Decode all selected files in parallel on Rayon worker threads
    let decoded: Result<Vec<(String, u32, u32, Vec<u8>)>, String> =
        tauri::async_runtime::spawn_blocking(move || {
            paths
                .par_iter()
                .map(|path_str| {
                    let path = Path::new(path_str);
                    let file_name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("Image")
                        .to_string();
                    let dyn_img = image::open(path)
                        .map_err(|e| format!("Failed to decode '{}': {e}", file_name))?;
                    let rgba_img = dyn_img.to_rgba8();
                    let (w, h) = rgba_img.dimensions();
                    Ok((file_name, w, h, rgba_img.into_raw()))
                })
                .collect()
        })
        .await
        .map_err(|e| format!("Task join error: {e}"))?;

    let decoded = decoded?;
    let mut new_stored = Vec::with_capacity(decoded.len());
    for (name, w, h, rgba) in decoded {
        let id = state.next_id.fetch_add(1, Ordering::Relaxed);
        new_stored.push(create_stored_image(id, name, w, h, rgba)?);
    }

    let mut images = state
        .images
        .lock()
        .map_err(|_| "Failed to lock image store".to_string())?;
    images.extend(new_stored);
    Ok(state.build_status(&images))
}

#[tauri::command]
async fn add_image_bytes(
    name: String,
    bytes: Vec<u8>,
    state: State<'_, AppState>,
) -> Result<StackStatusPayload, String> {
    let (w, h, rgba) = tauri::async_runtime::spawn_blocking(move || -> Result<(u32, u32, Vec<u8>), String> {
        let dyn_img =
            image::load_from_memory(&bytes).map_err(|e| format!("Failed to decode image: {e}"))?;
        let rgba_img = dyn_img.to_rgba8();
        let (w, h) = rgba_img.dimensions();
        Ok((w, h, rgba_img.into_raw()))
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))??;

    let id = state.next_id.fetch_add(1, Ordering::Relaxed);
    let stored = create_stored_image(id, name, w, h, rgba)?;

    let mut images = state
        .images
        .lock()
        .map_err(|_| "Failed to lock image store".to_string())?;
    images.push(stored);
    Ok(state.build_status(&images))
}

#[tauri::command]
fn remove_image(id: u64, state: State<'_, AppState>) -> Result<StackStatusPayload, String> {
    let mut images = state
        .images
        .lock()
        .map_err(|_| "Failed to lock image store".to_string())?;
    images.retain(|img| img.id != id);
    Ok(state.build_status(&images))
}

#[tauri::command]
fn reorder_images(
    ordered_ids: Vec<u64>,
    state: State<'_, AppState>,
) -> Result<StackStatusPayload, String> {
    let mut images = state
        .images
        .lock()
        .map_err(|_| "Failed to lock image store".to_string())?;

    let mut reordered = Vec::with_capacity(images.len());
    for target_id in ordered_ids {
        if let Some(pos) = images.iter().position(|img| img.id == target_id) {
            reordered.push(images.remove(pos));
        }
    }
    // Keep any remaining images that weren't in ordered_ids
    reordered.append(&mut images);
    *images = reordered;

    Ok(state.build_status(&images))
}

#[tauri::command]
fn clear_images(state: State<'_, AppState>) -> Result<StackStatusPayload, String> {
    let mut images = state
        .images
        .lock()
        .map_err(|_| "Failed to lock image store".to_string())?;
    images.clear();
    if let Ok(mut last) = state.last_result.lock() {
        *last = None;
    }
    Ok(state.build_status(&images))
}

#[tauri::command]
fn downscale_inputs(
    scale_factor: f32,
    state: State<'_, AppState>,
) -> Result<StackStatusPayload, String> {
    let scale = scale_factor.clamp(0.1, 0.95);
    let mut images = state
        .images
        .lock()
        .map_err(|_| "Failed to lock image store".to_string())?;

    images.par_iter_mut().for_each(|stored| {
        let new_w = ((stored.raw.width as f32 * scale).round() as u32).max(1);
        let new_h = ((stored.raw.height as f32 * scale).round() as u32).max(1);
        if let Some(src_img) =
            RgbaImage::from_raw(stored.raw.width, stored.raw.height, stored.raw.data.clone())
        {
            let resized = image::imageops::resize(&src_img, new_w, new_h, FilterType::Triangle);
            stored.raw =
                RawImage::with_known_alpha(new_w, new_h, resized.into_raw(), stored.raw.has_alpha);
        }
    });

    Ok(state.build_status(&images))
}

#[tauri::command]
fn cancel_stack(state: State<'_, AppState>) -> Result<(), String> {
    state.cancel_flag.store(true, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
async fn run_stack(
    app: AppHandle,
    options: StackOptions,
    state: State<'_, AppState>,
) -> Result<StackPreviewResponse, String> {
    // Persist user's chosen options
    {
        if let Ok(mut cfg) = state.config.lock() {
            cfg.last_mode = options.mode;
            cfg.auto_normalize = options.auto_normalize;
            cfg.size_match = options.size_match;
            cfg.fit_mode = options.fit_mode;
            let _ = cfg.save(&state.config_dir);
        }
    }

    let raw_images: Vec<RawImage> = {
        let images = state
            .images
            .lock()
            .map_err(|_| "Failed to lock image store".to_string())?;
        if images.len() < 2 {
            return Err("Add at least 2 images to compute a stack.".to_string());
        }
        images.iter().map(|s| s.raw.clone()).collect()
    };

    state.cancel_flag.store(false, Ordering::Relaxed);
    let cancel_flag = Arc::clone(&state.cancel_flag);
    let app_for_progress = app.clone();
    let opts_clone = options.clone();

    let stack_res: Result<StackResult, StackError> =
        tauri::async_runtime::spawn_blocking(move || {
            stack::stack_images(&raw_images, &opts_clone, Some(&cancel_flag), |pct| {
                let stage = if pct < 10 {
                    "Aligning & matching dimensions..."
                } else if pct < 95 {
                    "Computing pixel stack across CPU cores..."
                } else {
                    "Normalizing & preparing preview..."
                };
                let _ = app_for_progress.emit(
                    "stack-progress",
                    ProgressPayload {
                        percent: pct,
                        stage: stage.to_string(),
                    },
                );
            })
        })
        .await
        .map_err(|e| format!("Background worker failed: {e}"))?;

    let result = stack_res.map_err(|e| e.to_string())?;

    let (preview_data_url, preview_width, preview_height) =
        encode_png_data_url(result.width, result.height, &result.rgba, PREVIEW_MAX_DIM)?;

    let response = StackPreviewResponse {
        width: result.width,
        height: result.height,
        preview_width,
        preview_height,
        render_ms: result.render_ms,
        preview_data_url,
        mode_label: options.mode.label().to_string(),
    };

    // Store full-resolution result in Rust memory for lossless Copy / Save / Re-stack
    {
        let mut last = state
            .last_result
            .lock()
            .map_err(|_| "Failed to lock result buffer".to_string())?;
        *last = Some(result);
    }

    Ok(response)
}

#[tauri::command]
fn copy_result_to_clipboard(state: State<'_, AppState>) -> Result<(), String> {
    let last_guard = state
        .last_result
        .lock()
        .map_err(|_| "Failed to lock result buffer".to_string())?;
    let res = last_guard
        .as_ref()
        .ok_or_else(|| "No stacked result available to copy yet.".to_string())?;

    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| format!("Failed to open system clipboard: {e}"))?;

    let img_data = arboard::ImageData {
        width: res.width as usize,
        height: res.height as usize,
        bytes: Cow::Borrowed(&res.rgba),
    };

    clipboard
        .set_image(img_data)
        .map_err(|e| format!("Failed to copy image to clipboard: {e}"))?;

    Ok(())
}

#[tauri::command]
async fn save_result_image(
    path: String,
    format: String,
    jpeg_quality: u8,
    state: State<'_, AppState>,
) -> Result<SaveResultResponse, String> {
    let res_clone = {
        let last_guard = state
            .last_result
            .lock()
            .map_err(|_| "Failed to lock result buffer".to_string())?;
        last_guard
            .as_ref()
            .cloned()
            .ok_or_else(|| "No stacked result available to save.".to_string())?
    };

    // Update saved directory and JPEG quality preference
    {
        if let Ok(mut cfg) = state.config.lock() {
            cfg.jpeg_quality = jpeg_quality.clamp(1, 100);
            if let Some(parent) = Path::new(&path).parent() {
                cfg.last_save_dir = Some(parent.to_string_lossy().to_string());
            }
            let _ = cfg.save(&state.config_dir);
        }
    }

    let path_clone = path.clone();
    let fmt_lower = format.to_lowercase();
    let quality = jpeg_quality.clamp(1, 100);

    let bytes_written = tauri::async_runtime::spawn_blocking(move || -> Result<u64, String> {
        let out_path = Path::new(&path_clone);
        let file =
            File::create(out_path).map_err(|e| format!("Failed to create output file: {e}"))?;
        let mut writer = BufWriter::new(file);

        let ext = out_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or(&fmt_lower)
            .to_lowercase();

        let effective_fmt = if !fmt_lower.is_empty() {
            fmt_lower.as_str()
        } else {
            ext.as_str()
        };

        match effective_fmt {
            "jpg" | "jpeg" => {
                // Convert RGBA8 -> RGB8 (composite over dark background if transparent)
                let mut rgb = Vec::with_capacity((res_clone.width * res_clone.height * 3) as usize);
                for px in res_clone.rgba.chunks_exact(4) {
                    let a = px[3] as u16;
                    if a == 255 {
                        rgb.extend_from_slice(&px[0..3]);
                    } else {
                        let r = ((px[0] as u16 * a) / 255) as u8;
                        let g = ((px[1] as u16 * a) / 255) as u8;
                        let b = ((px[2] as u16 * a) / 255) as u8;
                        rgb.extend_from_slice(&[r, g, b]);
                    }
                }
                let encoder = JpegEncoder::new_with_quality(&mut writer, quality);
                encoder
                    .write_image(
                        &rgb,
                        res_clone.width,
                        res_clone.height,
                        ColorType::Rgb8.into(),
                    )
                    .map_err(|e| format!("JPEG encoding error: {e}"))?;
            }
            "webp" => {
                let encoder = WebPEncoder::new_lossless(&mut writer);
                encoder
                    .write_image(
                        &res_clone.rgba,
                        res_clone.width,
                        res_clone.height,
                        ColorType::Rgba8.into(),
                    )
                    .map_err(|e| format!("WebP encoding error: {e}"))?;
            }
            _ => {
                let encoder = PngEncoder::new(&mut writer);
                encoder
                    .write_image(
                        &res_clone.rgba,
                        res_clone.width,
                        res_clone.height,
                        ColorType::Rgba8.into(),
                    )
                    .map_err(|e| format!("PNG encoding error: {e}"))?;
            }
        }

        drop(writer);
        let meta = std::fs::metadata(out_path)
            .map_err(|e| format!("Failed to read saved file metadata: {e}"))?;
        Ok(meta.len())
    })
    .await
    .map_err(|e| format!("Save worker failed: {e}"))??;

    Ok(SaveResultResponse {
        path,
        bytes_written,
    })
}

#[tauri::command]
fn add_result_to_stack(state: State<'_, AppState>) -> Result<StackStatusPayload, String> {
    let res = {
        let last_guard = state
            .last_result
            .lock()
            .map_err(|_| "Failed to lock result buffer".to_string())?;
        last_guard
            .as_ref()
            .cloned()
            .ok_or_else(|| "No stacked result available to add.".to_string())?
    };

    let id = state.next_id.fetch_add(1, Ordering::Relaxed);
    let name = format!("Stacked Result #{id}");
    let stored = create_stored_image(id, name, res.width, res.height, res.rgba)?;

    let mut images = state
        .images
        .lock()
        .map_err(|_| "Failed to lock image store".to_string())?;
    images.push(stored);
    Ok(state.build_status(&images))
}

#[tauri::command]
fn save_window_config(
    width: u32,
    height: u32,
    state: State<'_, AppState>,
) -> Result<(), String> {
    if width < 400 || height < 300 {
        return Ok(());
    }
    let mut cfg = state
        .config
        .lock()
        .map_err(|_| "Failed to lock config".to_string())?;
    cfg.window_width = width;
    cfg.window_height = height;
    cfg.save(&state.config_dir)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            let config_dir = app
                .path()
                .app_config_dir()
                .unwrap_or_else(|_| PathBuf::from("."));
            let app_state = AppState::new(config_dir);

            // Restore persisted window dimensions if available
            if let Ok(cfg) = app_state.config.lock() {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.set_size(tauri::Size::Logical(tauri::LogicalSize {
                        width: cfg.window_width as f64,
                        height: cfg.window_height as f64,
                    }));
                }
            }

            app.manage(app_state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_initial_state,
            paste_clipboard_image,
            add_image_files,
            add_image_bytes,
            remove_image,
            reorder_images,
            clear_images,
            downscale_inputs,
            run_stack,
            cancel_stack,
            copy_result_to_clipboard,
            save_result_image,
            add_result_to_stack,
            save_window_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running StackShot Tauri application");
}

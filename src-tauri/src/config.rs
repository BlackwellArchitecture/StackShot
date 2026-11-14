//! Persistent application configuration stored as a lightweight JSON file
//! in the OS-specific application config directory.

use crate::stack::{FitMode, SizeMatchMode, StackMode};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub last_mode: StackMode,
    pub auto_normalize: bool,
    pub size_match: SizeMatchMode,
    pub fit_mode: FitMode,
    pub jpeg_quality: u8,
    pub last_save_dir: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            window_width: 1280,
            window_height: 820,
            last_mode: StackMode::Median,
            auto_normalize: true,
            size_match: SizeMatchMode::First,
            fit_mode: FitMode::CenterCrop,
            jpeg_quality: 92,
            last_save_dir: None,
        }
    }
}

impl AppConfig {
    pub fn config_file_path(app_data_dir: &Path) -> PathBuf {
        app_data_dir.join("settings.json")
    }

    /// Loads config from `<app_data_dir>/settings.json`, falling back to defaults if missing or invalid.
    pub fn load(app_data_dir: &Path) -> Self {
        let path = Self::config_file_path(app_data_dir);
        match fs::read_to_string(&path) {
            Ok(raw) => serde_json::from_str::<AppConfig>(&raw).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Saves config atomically to `<app_data_dir>/settings.json`.
    pub fn save(&self, app_data_dir: &Path) -> Result<(), String> {
        if let Err(e) = fs::create_dir_all(app_data_dir) {
            return Err(format!("Failed to create app config dir: {e}"));
        }
        let path = Self::config_file_path(app_data_dir);
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {e}"))?;
        fs::write(&path, json).map_err(|e| format!("Failed to write config file: {e}"))
    }
}

export type StackMode =
  | "median"
  | "mean"
  | "maximum"
  | "minimum"
  | "range"
  | "std_dev"
  | "summation"
  | "variance"
  | "skewness"
  | "kurtosis"
  | "entropy";

export type SizeMatchMode = "first" | "largest";

export type FitMode = "center_crop" | "letterbox";

export type ExportFormat = "png" | "jpeg" | "webp";

export interface StackOptions {
  mode: StackMode;
  auto_normalize: boolean;
  size_match: SizeMatchMode;
  fit_mode: FitMode;
}

export interface AppConfig {
  window_width: number;
  window_height: number;
  last_mode: StackMode;
  auto_normalize: boolean;
  size_match: SizeMatchMode;
  fit_mode: FitMode;
  jpeg_quality: number;
  last_save_dir: string | null;
}

export interface ImageMeta {
  id: number;
  name: string;
  width: number;
  height: number;
  has_alpha: boolean;
  byte_size: number;
  thumbnail_data_url: string;
}

export interface StackStatusPayload {
  images: ImageMeta[];
  total_memory_bytes: number;
  estimated_peak_bytes: number;
  memory_warning: boolean;
}

export interface ModeInfo {
  id: StackMode;
  label: string;
  description: string;
  is_low_output: boolean;
}

export interface InitialStatePayload {
  config: AppConfig;
  modes: ModeInfo[];
  status: StackStatusPayload;
}

export interface ProgressPayload {
  percent: number;
  stage: string;
}

export interface StackPreviewResponse {
  width: number;
  height: number;
  preview_width: number;
  preview_height: number;
  render_ms: number;
  preview_data_url: string;
  mode_label: string;
}

export interface SaveResultResponse {
  path: string;
  bytes_written: number;
}

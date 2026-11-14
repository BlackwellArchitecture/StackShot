import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open, save } from "@tauri-apps/plugin-dialog";
import type {
  ExportFormat,
  FitMode,
  ImageMeta,
  InitialStatePayload,
  ModeInfo,
  ProgressPayload,
  SaveResultResponse,
  SizeMatchMode,
  StackMode,
  StackOptions,
  StackPreviewResponse,
  StackStatusPayload,
} from "./types";

const DEFAULT_MODES: ModeInfo[] = [
  {
    id: "median",
    label: "Median",
    description:
      "Picks the middle value per pixel — ideal for removing moving objects, tourists, traffic, or transient noise.",
    is_low_output: false,
  },
  {
    id: "mean",
    label: "Mean (Average)",
    description:
      "Averages pixel values across frames — simulates long exposures, smooths water/clouds, and reduces sensor noise.",
    is_low_output: false,
  },
  {
    id: "maximum",
    label: "Maximum",
    description:
      "Keeps the brightest pixel value — great for star trails, fireworks, light painting, and lightning composites.",
    is_low_output: false,
  },
  {
    id: "minimum",
    label: "Minimum",
    description:
      "Keeps the darkest pixel value — removes bright flashes, specular glare, or light-colored moving objects.",
    is_low_output: false,
  },
  {
    id: "range",
    label: "Range",
    description:
      "Computes Maximum minus Minimum — highlights total motion amplitude and checks for tripod shift.",
    is_low_output: true,
  },
  {
    id: "std_dev",
    label: "Standard Deviation",
    description:
      "Measures variation from the mean — used in scientific imaging to isolate movement and temporal change.",
    is_low_output: true,
  },
  {
    id: "summation",
    label: "Summation",
    description:
      "Adds pixel intensities (clamped to 255) — boosts faint signals in astrophotography and low-light captures.",
    is_low_output: false,
  },
  {
    id: "variance",
    label: "Variance",
    description:
      "Squared deviation from the mean — accentuates high-turbulence regions and strong temporal fluctuations.",
    is_low_output: true,
  },
  {
    id: "skewness",
    label: "Skewness",
    description:
      "Measures asymmetry of pixel distribution over time — detects brief bright or dark directional anomalies.",
    is_low_output: true,
  },
  {
    id: "kurtosis",
    label: "Kurtosis",
    description:
      "Measures outlier extremity (tailedness) across frames — isolates rare, sudden spikes against steady backgrounds.",
    is_low_output: true,
  },
  {
    id: "entropy",
    label: "Entropy",
    description:
      "Computes Shannon information entropy across frames — maps temporal randomness and visual complexity.",
    is_low_output: true,
  },
];

class StackShotApp {
  private modes: ModeInfo[] = DEFAULT_MODES;
  private images: ImageMeta[] = [];
  private selectedImageId: number | null = null;
  private draggedImageId: number | null = null;

  private activeMode: StackMode = "median";
  private autoNormalize = true;
  private sizeMatch: SizeMatchMode = "first";
  private fitMode: FitMode = "center_crop";
  private exportFormat: ExportFormat = "png";
  private jpegQuality = 92;
  private lastSaveDir: string | null = null;

  private isStacking = false;
  private hasResult = false;
  private lastPreview: StackPreviewResponse | null = null;
  private dismissedMemWarning = false;

  // Viewport Zoom & Pan state
  private zoom = 1;
  private panX = 0;
  private panY = 0;
  private isPanning = false;
  private panStartX = 0;
  private panStartY = 0;

  // DOM Elements
  private elThumbStrip = document.getElementById("thumb-strip") as HTMLElement;
  private elThumbEmptyHint = document.getElementById("thumb-empty-hint") as HTMLElement;
  private elImageCounter = document.getElementById("image-counter") as HTMLElement;
  private elMemoryCounter = document.getElementById("memory-counter") as HTMLElement;
  private elMemBanner = document.getElementById("memory-warning-banner") as HTMLElement;
  private elMemDetail = document.getElementById("memory-warning-detail") as HTMLElement;

  private elModeGrid = document.getElementById("mode-grid") as HTMLElement;
  private elActiveModePill = document.getElementById("active-mode-pill") as HTMLElement;
  private elModeDesc = document.getElementById("mode-description") as HTMLElement;
  private elAutoNormRow = document.getElementById("auto-normalize-row") as HTMLElement;
  private elChkAutoNorm = document.getElementById("chk-auto-normalize") as HTMLInputElement;

  private elSelSizeMatch = document.getElementById("sel-size-match") as HTMLSelectElement;
  private elSelFitMode = document.getElementById("sel-fit-mode") as HTMLSelectElement;

  private elBtnStack = document.getElementById("btn-stack") as HTMLButtonElement;
  private elMinImagesHint = document.getElementById("min-images-hint") as HTMLElement;
  private elProgressContainer = document.getElementById("progress-container") as HTMLElement;
  private elProgressStage = document.getElementById("progress-stage") as HTMLElement;
  private elProgressPct = document.getElementById("progress-pct") as HTMLElement;
  private elProgressFill = document.getElementById("progress-fill") as HTMLElement;

  private elBtnCopyResult = document.getElementById("btn-copy-result") as HTMLButtonElement;
  private elBtnSaveResult = document.getElementById("btn-save-result") as HTMLButtonElement;
  private elBtnAddResultToStack = document.getElementById("btn-add-result-to-stack") as HTMLButtonElement;
  private elSelExportFormat = document.getElementById("sel-export-format") as HTMLSelectElement;
  private elJpegQualityRow = document.getElementById("jpeg-quality-row") as HTMLElement;
  private elRngJpegQuality = document.getElementById("rng-jpeg-quality") as HTMLInputElement;
  private elLblJpegQuality = document.getElementById("lbl-jpeg-quality") as HTMLElement;

  private elEmptyDropzone = document.getElementById("empty-dropzone") as HTMLElement;
  private elViewportContainer = document.getElementById("viewport-container") as HTMLElement;
  private elViewportStage = document.getElementById("viewport-stage") as HTMLElement;
  private elPreviewImg = document.getElementById("preview-img") as HTMLImageElement;
  private elDragOverlay = document.getElementById("drag-overlay") as HTMLElement;

  private elStatMode = document.getElementById("stat-mode") as HTMLElement;
  private elStatDimensions = document.getElementById("stat-dimensions") as HTMLElement;
  private elStatRenderTime = document.getElementById("stat-render-time") as HTMLElement;
  private elLblZoomLevel = document.getElementById("lbl-zoom-level") as HTMLElement;

  private elFileInputFallback = document.getElementById("file-input-fallback") as HTMLInputElement;
  private elToastContainer = document.getElementById("toast-container") as HTMLElement;

  public async init(): Promise<void> {
    this.renderModeButtons();
    this.bindUIEvents();
    this.bindViewportEvents();
    this.bindKeyboardShortcuts();
    await this.bindTauriEvents();
    await this.loadInitialState();
  }

  private async loadInitialState(): Promise<void> {
    try {
      const init = await invoke<InitialStatePayload>("get_initial_state");
      if (init.modes && init.modes.length > 0) {
        this.modes = init.modes;
      }
      this.activeMode = init.config.last_mode || "median";
      this.autoNormalize = init.config.auto_normalize ?? true;
      this.sizeMatch = init.config.size_match || "first";
      this.fitMode = init.config.fit_mode || "center_crop";
      this.jpegQuality = init.config.jpeg_quality || 92;
      this.lastSaveDir = init.config.last_save_dir;

      this.elChkAutoNorm.checked = this.autoNormalize;
      this.elSelSizeMatch.value = this.sizeMatch;
      this.elSelFitMode.value = this.fitMode;
      this.elRngJpegQuality.value = String(this.jpegQuality);
      this.elLblJpegQuality.textContent = `${this.jpegQuality}%`;

      this.renderModeButtons();
      this.updateStatus(init.status);
    } catch {
      // Running outside Tauri native shell (e.g. Vite browser preview)
      this.renderModeButtons();
    }
  }

  private renderModeButtons(): void {
    this.elModeGrid.innerHTML = "";
    for (const mode of this.modes) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `mode-btn${mode.id === this.activeMode ? " active" : ""}`;
      btn.textContent = mode.label;
      btn.setAttribute("role", "radio");
      btn.setAttribute("aria-checked", String(mode.id === this.activeMode));
      btn.addEventListener("click", () => {
        this.selectMode(mode.id);
      });
      this.elModeGrid.appendChild(btn);
    }
    this.syncActiveModeUI();
  }

  private selectMode(modeId: StackMode): void {
    this.activeMode = modeId;
    const buttons = this.elModeGrid.querySelectorAll(".mode-btn");
    this.modes.forEach((m, idx) => {
      const btn = buttons[idx];
      if (btn) {
        btn.classList.toggle("active", m.id === modeId);
        btn.setAttribute("aria-checked", String(m.id === modeId));
      }
    });
    this.syncActiveModeUI();
  }

  private syncActiveModeUI(): void {
    const info = this.modes.find((m) => m.id === this.activeMode) || this.modes[0];
    this.elActiveModePill.textContent = info.label;
    this.elModeDesc.textContent = info.description;
    this.elAutoNormRow.classList.toggle("hidden", !info.is_low_output);
  }

  private bindUIEvents(): void {
    // Window Titlebar Controls
    document.getElementById("btn-win-min")?.addEventListener("click", async () => {
      try {
        await getCurrentWindow().minimize();
      } catch {}
    });
    document.getElementById("btn-win-max")?.addEventListener("click", async () => {
      try {
        await getCurrentWindow().toggleMaximize();
      } catch {}
    });
    document.getElementById("btn-win-close")?.addEventListener("click", async () => {
      try {
        await getCurrentWindow().close();
      } catch {}
    });

    // Paste & Add Images
    document.getElementById("btn-paste-image")?.addEventListener("click", () => this.pasteFromClipboard());
    document.getElementById("btn-empty-paste")?.addEventListener("click", () => this.pasteFromClipboard());
    document.getElementById("btn-add-images")?.addEventListener("click", () => this.openFileDialog());
    document.getElementById("btn-empty-browse")?.addEventListener("click", () => this.openFileDialog());
    document.getElementById("btn-clear-all")?.addEventListener("click", () => this.clearAllImages());

    // Memory Warning Controls
    document.getElementById("btn-downscale-50")?.addEventListener("click", () => this.downscaleInputs(0.5));
    document.getElementById("btn-dismiss-mem-warn")?.addEventListener("click", () => {
      this.dismissedMemWarning = true;
      this.elMemBanner.classList.add("hidden");
    });

    // Stack Options
    this.elChkAutoNorm.addEventListener("change", () => {
      this.autoNormalize = this.elChkAutoNorm.checked;
    });
    this.elSelSizeMatch.addEventListener("change", () => {
      this.sizeMatch = this.elSelSizeMatch.value as SizeMatchMode;
    });
    this.elSelFitMode.addEventListener("change", () => {
      this.fitMode = this.elSelFitMode.value as FitMode;
    });

    // Stack & Cancel Buttons
    this.elBtnStack.addEventListener("click", () => this.runStack());
    document.getElementById("btn-cancel-stack")?.addEventListener("click", () => this.cancelStack());

    // Export & Result Actions
    this.elBtnCopyResult.addEventListener("click", () => this.copyResult());
    this.elBtnSaveResult.addEventListener("click", () => this.saveResult());
    this.elBtnAddResultToStack.addEventListener("click", () => this.addResultBackToStack());

    this.elSelExportFormat.addEventListener("change", () => {
      this.exportFormat = this.elSelExportFormat.value as ExportFormat;
      this.elJpegQualityRow.classList.toggle("hidden", this.exportFormat !== "jpeg");
    });

    this.elRngJpegQuality.addEventListener("input", () => {
      this.jpegQuality = Number(this.elRngJpegQuality.value);
      this.elLblJpegQuality.textContent = `${this.jpegQuality}%`;
    });

    // Fallback browser file input
    this.elFileInputFallback.addEventListener("change", async () => {
      if (this.elFileInputFallback.files && this.elFileInputFallback.files.length > 0) {
        await this.handleBrowserFiles(Array.from(this.elFileInputFallback.files));
        this.elFileInputFallback.value = "";
      }
    });

    // HTML5 Drag & Drop on window
    window.addEventListener("dragover", (e) => {
      if (this.draggedImageId !== null) return; // Internal thumbnail reorder
      e.preventDefault();
      this.elDragOverlay.classList.remove("hidden");
    });
    window.addEventListener("dragleave", (e) => {
      if (e.relatedTarget === null) {
        this.elDragOverlay.classList.add("hidden");
      }
    });
    window.addEventListener("drop", async (e) => {
      if (this.draggedImageId !== null) return;
      e.preventDefault();
      this.elDragOverlay.classList.add("hidden");
      if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
        await this.handleBrowserFiles(Array.from(e.dataTransfer.files));
      }
    });
  }

  private async bindTauriEvents(): Promise<void> {
    try {
      await listen<ProgressPayload>("stack-progress", (event) => {
        const { percent, stage } = event.payload;
        this.elProgressPct.textContent = `${percent}%`;
        this.elProgressStage.textContent = stage;
        this.elProgressFill.style.width = `${percent}%`;
      });

      // Tauri native file drop event
      await listen<{ paths: string[] }>("tauri://drag-drop", async (event) => {
        this.elDragOverlay.classList.add("hidden");
        if (event.payload?.paths && event.payload.paths.length > 0) {
          await this.addImagePaths(event.payload.paths);
        }
      });

      // Persist window size changes (debounced)
      let resizeTimer: number | undefined;
      window.addEventListener("resize", () => {
        window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(() => {
          invoke("save_window_config", {
            width: Math.round(window.innerWidth),
            height: Math.round(window.innerHeight),
          }).catch(() => {});
        }, 400);
      });
    } catch {
      // Ignore if running in plain browser mode
    }
  }

  private bindKeyboardShortcuts(): void {
    window.addEventListener("keydown", (e) => {
      const isCtrlOrCmd = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();

      // Ctrl/Cmd + V: Paste from clipboard
      if (isCtrlOrCmd && key === "v") {
        e.preventDefault();
        this.pasteFromClipboard();
        return;
      }

      // Ctrl/Cmd + C: Copy full-resolution result
      if (isCtrlOrCmd && key === "c") {
        if (this.hasResult) {
          e.preventDefault();
          this.copyResult();
        }
        return;
      }

      // Ctrl/Cmd + S: Save result
      if (isCtrlOrCmd && key === "s") {
        if (this.hasResult) {
          e.preventDefault();
          this.saveResult();
        }
        return;
      }

      // Enter: Run Stack
      if (e.key === "Enter" && !isCtrlOrCmd) {
        const activeTag = document.activeElement?.tagName.toLowerCase();
        if (activeTag !== "select" && activeTag !== "input") {
          e.preventDefault();
          if (!this.isStacking) {
            this.runStack();
          }
        }
        return;
      }

      // Escape: Cancel active render
      if (e.key === "Escape") {
        if (this.isStacking) {
          e.preventDefault();
          this.cancelStack();
        }
        return;
      }

      // Delete or Backspace: Remove selected thumbnail
      if (e.key === "Delete" || e.key === "Backspace") {
        const activeTag = document.activeElement?.tagName.toLowerCase();
        if (activeTag !== "input" && activeTag !== "select" && this.selectedImageId !== null) {
          e.preventDefault();
          this.removeImage(this.selectedImageId);
        }
      }
    });
  }

  private bindViewportEvents(): void {
    // Scroll wheel zoom around cursor
    this.elViewportContainer.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const rect = this.elViewportStage.getBoundingClientRect();
        const cursorX = e.clientX - rect.left;
        const cursorY = e.clientY - rect.top;

        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const nextZoom = Math.min(16, Math.max(0.08, this.zoom * factor));
        const scaleChange = nextZoom / this.zoom;

        this.panX = cursorX - (cursorX - this.panX) * scaleChange;
        this.panY = cursorY - (cursorY - this.panY) * scaleChange;
        this.zoom = nextZoom;
        this.applyViewportTransform();
      },
      { passive: false }
    );

    // Drag to pan
    this.elViewportContainer.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || (e.target as HTMLElement).closest(".viewport-hud")) return;
      this.isPanning = true;
      this.panStartX = e.clientX - this.panX;
      this.panStartY = e.clientY - this.panY;
      this.elViewportContainer.classList.add("panning");
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isPanning) return;
      this.panX = e.clientX - this.panStartX;
      this.panY = e.clientY - this.panStartY;
      this.applyViewportTransform();
    });

    window.addEventListener("mouseup", () => {
      if (this.isPanning) {
        this.isPanning = false;
        this.elViewportContainer.classList.remove("panning");
      }
    });

    // Double-click to fit to window
    this.elViewportStage.addEventListener("dblclick", () => {
      this.fitPreviewToWindow();
    });

    // HUD Zoom Buttons
    document.getElementById("btn-zoom-in")?.addEventListener("click", () => {
      this.zoomAroundCenter(1.25);
    });
    document.getElementById("btn-zoom-out")?.addEventListener("click", () => {
      this.zoomAroundCenter(0.8);
    });
    document.getElementById("btn-zoom-100")?.addEventListener("click", () => {
      this.setActualPixelZoom();
    });
    document.getElementById("btn-zoom-fit")?.addEventListener("click", () => {
      this.fitPreviewToWindow();
    });
  }

  private zoomAroundCenter(factor: number): void {
    const rect = this.elViewportStage.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const nextZoom = Math.min(16, Math.max(0.08, this.zoom * factor));
    const ratio = nextZoom / this.zoom;
    this.panX = cx - (cx - this.panX) * ratio;
    this.panY = cy - (cy - this.panY) * ratio;
    this.zoom = nextZoom;
    this.applyViewportTransform();
  }

  private fitPreviewToWindow(): void {
    if (!this.lastPreview) return;
    const rect = this.elViewportStage.getBoundingClientRect();
    const pad = 48;
    const availW = Math.max(100, rect.width - pad);
    const availH = Math.max(100, rect.height - pad);
    const imgW = this.lastPreview.preview_width;
    const imgH = this.lastPreview.preview_height;

    this.zoom = Math.min(availW / imgW, availH / imgH, 1.0);
    this.panX = Math.round((rect.width - imgW * this.zoom) / 2);
    this.panY = Math.round((rect.height - imgH * this.zoom) / 2);
    this.applyViewportTransform();
  }

  private setActualPixelZoom(): void {
    if (!this.lastPreview) return;
    const rect = this.elViewportStage.getBoundingClientRect();
    // Scale relative to full resolution
    const fullRatio = this.lastPreview.width / this.lastPreview.preview_width;
    this.zoom = fullRatio;
    this.panX = Math.round((rect.width - this.lastPreview.preview_width * this.zoom) / 2);
    this.panY = Math.round((rect.height - this.lastPreview.preview_height * this.zoom) / 2);
    this.applyViewportTransform();
  }

  private applyViewportTransform(): void {
    this.elPreviewImg.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
    if (this.lastPreview) {
      const displayScale =
        (this.zoom * this.lastPreview.preview_width) / Math.max(1, this.lastPreview.width);
      this.elLblZoomLevel.textContent = `${Math.round(displayScale * 100)}%`;
    } else {
      this.elLblZoomLevel.textContent = `${Math.round(this.zoom * 100)}%`;
    }
  }

  public async pasteFromClipboard(): Promise<void> {
    try {
      const status = await invoke<StackStatusPayload>("paste_clipboard_image");
      this.updateStatus(status);
      const newest = status.images[status.images.length - 1];
      if (newest) {
        this.selectedImageId = newest.id;
        this.renderThumbnails();
        this.showToast(`Pasted ${newest.name} (${newest.width}×${newest.height})`, "success");
      }
    } catch (err) {
      this.showToast(String(err), "error");
    }
  }

  public async openFileDialog(): Promise<void> {
    try {
      const selected = await open({
        multiple: true,
        title: "Select Images to Stack",
        filters: [
          {
            name: "Images",
            extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif", "ico"],
          },
        ],
      });

      if (!selected) return;
      const paths = Array.isArray(selected) ? selected : [selected];
      if (paths.length > 0) {
        await this.addImagePaths(paths);
      }
    } catch {
      // Fallback to standard file input when running in browser preview
      this.elFileInputFallback.click();
    }
  }

  private async addImagePaths(paths: string[]): Promise<void> {
    try {
      const prevCount = this.images.length;
      const status = await invoke<StackStatusPayload>("add_image_files", { paths });
      this.updateStatus(status);
      const added = status.images.length - prevCount;
      if (added > 0) {
        this.showToast(`Added ${added} image${added === 1 ? "" : "s"} to stack`, "success");
      }
    } catch (err) {
      this.showToast(String(err), "error");
    }
  }

  private async handleBrowserFiles(files: File[]): Promise<void> {
    let added = 0;
    for (const file of files) {
      try {
        const buf = await file.arrayBuffer();
        const bytes = Array.from(new Uint8Array(buf));
        const status = await invoke<StackStatusPayload>("add_image_bytes", {
          name: file.name,
          bytes,
        });
        this.updateStatus(status);
        added++;
      } catch (err) {
        this.showToast(`Could not load ${file.name}: ${err}`, "error");
      }
    }
    if (added > 0) {
      this.showToast(`Added ${added} image${added === 1 ? "" : "s"} to stack`, "success");
    }
  }

  private async removeImage(id: number): Promise<void> {
    try {
      const status = await invoke<StackStatusPayload>("remove_image", { id });
      if (this.selectedImageId === id) {
        this.selectedImageId = status.images.length > 0 ? status.images[0].id : null;
      }
      this.updateStatus(status);
      this.showToast("Removed image from stack", "info");
    } catch (err) {
      this.showToast(String(err), "error");
    }
  }

  private async clearAllImages(): Promise<void> {
    try {
      const status = await invoke<StackStatusPayload>("clear_images");
      this.selectedImageId = null;
      this.hasResult = false;
      this.lastPreview = null;
      this.dismissedMemWarning = false;
      this.updateStatus(status);
      this.syncResultUI();
      this.showToast("Cleared all images", "info");
    } catch (err) {
      this.showToast(String(err), "error");
    }
  }

  private async downscaleInputs(scaleFactor: number): Promise<void> {
    try {
      const status = await invoke<StackStatusPayload>("downscale_inputs", {
        scaleFactor,
      });
      this.updateStatus(status);
      this.showToast("Downscaled inputs by 50% to optimize memory", "success");
    } catch (err) {
      this.showToast(String(err), "error");
    }
  }

  private updateStatus(status: StackStatusPayload): void {
    this.images = status.images;
    const count = this.images.length;
    this.elImageCounter.textContent = `${count} image${count === 1 ? "" : "s"}`;

    const mb = (status.total_memory_bytes / (1024 * 1024)).toFixed(1);
    this.elMemoryCounter.textContent = `${mb} MB in RAM`;

    const clearBtn = document.getElementById("btn-clear-all") as HTMLButtonElement | null;
    if (clearBtn) {
      clearBtn.disabled = count === 0;
    }

    // Require at least 2 images
    const canStack = count >= 2 && !this.isStacking;
    this.elBtnStack.disabled = !canStack;
    this.elMinImagesHint.classList.toggle("hidden", count >= 2);
    if (count === 1) {
      this.elMinImagesHint.textContent = "Add 1 more image to compute a stack.";
    } else if (count === 0) {
      this.elMinImagesHint.textContent = "Add at least 2 images to enable stacking.";
    }

    // Memory warning check
    if (status.memory_warning && !this.dismissedMemWarning) {
      const estGb = (status.estimated_peak_bytes / (1024 * 1024 * 1024)).toFixed(2);
      this.elMemDetail.textContent = `Estimated peak buffer usage is ~${estGb} GB across ${count} high-res images.`;
      this.elMemBanner.classList.remove("hidden");
    } else {
      this.elMemBanner.classList.add("hidden");
    }

    this.renderThumbnails();
  }

  private renderThumbnails(): void {
    // Remove existing cards
    const existingCards = this.elThumbStrip.querySelectorAll(".thumb-card");
    existingCards.forEach((c) => c.remove());

    if (this.images.length === 0) {
      this.elThumbEmptyHint.classList.remove("hidden");
      return;
    }
    this.elThumbEmptyHint.classList.add("hidden");

    this.images.forEach((img, index) => {
      const card = document.createElement("div");
      card.className = `thumb-card${img.id === this.selectedImageId ? " selected" : ""}`;
      card.draggable = true;
      card.dataset.id = String(img.id);
      card.title = `${img.name} (${img.width}×${img.height})`;

      card.innerHTML = `
        <div class="thumb-img-wrap">
          <img class="thumb-img" src="${img.thumbnail_data_url}" alt="${img.name}" />
        </div>
        <div class="thumb-meta">
          <span class="thumb-index">#${index + 1}</span>
          <span class="thumb-dims">${img.width}×${img.height}</span>
        </div>
        <button type="button" class="thumb-remove" title="Remove image (Delete)" aria-label="Remove ${img.name}">×</button>
      `;

      // Select thumbnail on click
      card.addEventListener("click", (e) => {
        if ((e.target as HTMLElement).closest(".thumb-remove")) return;
        this.selectedImageId = img.id;
        this.renderThumbnails();
      });

      // Remove button
      const removeBtn = card.querySelector(".thumb-remove") as HTMLButtonElement;
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.removeImage(img.id);
      });

      // Drag-to-reorder handlers
      card.addEventListener("dragstart", (e) => {
        this.draggedImageId = img.id;
        card.classList.add("dragging");
        e.dataTransfer?.setData("text/plain", String(img.id));
      });

      card.addEventListener("dragend", () => {
        this.draggedImageId = null;
        card.classList.remove("dragging");
        this.elThumbStrip
          .querySelectorAll(".thumb-card")
          .forEach((el) => el.classList.remove("drag-over"));
      });

      card.addEventListener("dragover", (e) => {
        if (this.draggedImageId === null || this.draggedImageId === img.id) return;
        e.preventDefault();
        card.classList.add("drag-over");
      });

      card.addEventListener("dragleave", () => {
        card.classList.remove("drag-over");
      });

      card.addEventListener("drop", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        card.classList.remove("drag-over");
        if (this.draggedImageId === null || this.draggedImageId === img.id) return;

        const orderedIds = this.images.map((item) => item.id);
        const fromIdx = orderedIds.indexOf(this.draggedImageId);
        const toIdx = orderedIds.indexOf(img.id);
        if (fromIdx !== -1 && toIdx !== -1) {
          orderedIds.splice(fromIdx, 1);
          orderedIds.splice(toIdx, 0, this.draggedImageId);
          try {
            const status = await invoke<StackStatusPayload>("reorder_images", {
              orderedIds,
            });
            this.updateStatus(status);
          } catch (err) {
            this.showToast(String(err), "error");
          }
        }
        this.draggedImageId = null;
      });

      this.elThumbStrip.appendChild(card);
    });
  }

  public async runStack(): Promise<void> {
    if (this.images.length < 2) {
      this.showToast("Please add at least 2 images before stacking.", "info");
      return;
    }
    if (this.isStacking) return;

    this.isStacking = true;
    this.elBtnStack.disabled = true;
    this.elProgressContainer.classList.remove("hidden");
    this.elProgressFill.style.width = "2%";
    this.elProgressPct.textContent = "2%";
    this.elProgressStage.textContent = "Starting stack computation...";

    const options: StackOptions = {
      mode: this.activeMode,
      auto_normalize: this.autoNormalize,
      size_match: this.sizeMatch,
      fit_mode: this.fitMode,
    };

    try {
      const res = await invoke<StackPreviewResponse>("run_stack", { options });
      this.lastPreview = res;
      this.hasResult = true;

      this.elPreviewImg.onload = () => {
        this.fitPreviewToWindow();
      };
      this.elPreviewImg.src = res.preview_data_url;

      this.elStatMode.textContent = res.mode_label;
      this.elStatDimensions.textContent = `${res.width} × ${res.height} px`;
      this.elStatRenderTime.textContent = `${res.render_ms.toFixed(1)} ms`;

      this.syncResultUI();
      this.showToast(
        `Rendered ${res.mode_label} stack (${res.width}×${res.height}) in ${res.render_ms.toFixed(1)} ms`,
        "success"
      );
    } catch (err) {
      const msg = String(err);
      if (msg.toLowerCase().includes("cancelled")) {
        this.showToast("Stack render cancelled", "info");
      } else {
        this.showToast(msg, "error");
      }
    } finally {
      this.isStacking = false;
      this.elBtnStack.disabled = this.images.length < 2;
      this.elProgressContainer.classList.add("hidden");
    }
  }

  public async cancelStack(): Promise<void> {
    if (!this.isStacking) return;
    try {
      await invoke("cancel_stack");
    } catch {}
  }

  private syncResultUI(): void {
    this.elBtnCopyResult.disabled = !this.hasResult;
    this.elBtnSaveResult.disabled = !this.hasResult;
    this.elBtnAddResultToStack.disabled = !this.hasResult;

    this.elEmptyDropzone.classList.toggle("hidden", this.hasResult);
    this.elViewportContainer.classList.toggle("hidden", !this.hasResult);
  }

  public async copyResult(): Promise<void> {
    if (!this.hasResult) return;
    try {
      await invoke("copy_result_to_clipboard");
      this.showToast("Copied full-resolution result to clipboard", "success");
    } catch (err) {
      this.showToast(String(err), "error");
    }
  }

  public async saveResult(): Promise<void> {
    if (!this.hasResult) return;
    const ext = this.exportFormat === "jpeg" ? "jpg" : this.exportFormat;
    const defaultName = `StackShot_${this.activeMode}.${ext}`;

    try {
      const targetPath = await save({
        title: "Save Stacked Image",
        defaultPath: this.lastSaveDir ? `${this.lastSaveDir}/${defaultName}` : defaultName,
        filters: [
          {
            name: this.exportFormat.toUpperCase(),
            extensions: ext === "jpg" ? ["jpg", "jpeg"] : [ext],
          },
        ],
      });

      if (!targetPath) return;

      const saved = await invoke<SaveResultResponse>("save_result_image", {
        path: targetPath,
        format: this.exportFormat,
        jpegQuality: this.jpegQuality,
      });

      const kb = (saved.bytes_written / 1024).toFixed(1);
      this.showToast(`Saved ${saved.path} (${kb} KB)`, "success");
    } catch (err) {
      this.showToast(String(err), "error");
    }
  }

  public async addResultBackToStack(): Promise<void> {
    if (!this.hasResult) return;
    try {
      const status = await invoke<StackStatusPayload>("add_result_to_stack");
      this.updateStatus(status);
      this.showToast("Added rendered result back to input stack", "success");
    } catch (err) {
      this.showToast(String(err), "error");
    }
  }

  private showToast(message: string, type: "info" | "success" | "error" = "info"): void {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    this.elToastContainer.appendChild(toast);

    window.setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transition = "opacity 150ms ease";
      window.setTimeout(() => toast.remove(), 160);
    }, 3200);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new StackShotApp();
  app.init();
});

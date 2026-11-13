export interface CameraResolution {
  width: number;
  height: number;
  fps: number;
}

export interface CameraDeviceSummary {
  deviceId: string;
  label: string;
}

export type CameraLossHandler = (reasonFa: string) => void;

export class CameraManager {
  private stream: MediaStream | null = null;
  private videoEl: HTMLVideoElement | null = null;
  private activeDeviceId: string | null = null;
  private resolution: CameraResolution = { width: 1280, height: 720, fps: 30 };
  private onCameraLost: CameraLossHandler | null = null;

  setLossHandler(handler: CameraLossHandler | null): void {
    this.onCameraLost = handler;
  }

  get currentResolution(): CameraResolution {
    return this.resolution;
  }

  get currentDeviceId(): string | null {
    return this.activeDeviceId;
  }

  get isStreaming(): boolean {
    return (
      this.stream !== null &&
      this.stream.getVideoTracks().some((track) => track.readyState === 'live')
    );
  }

  async listCameras(): Promise<CameraDeviceSummary[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((d) => d.kind === 'videoinput')
      .map((d, index) => ({
        deviceId: d.deviceId,
        label: d.label || `دوربین ${index + 1}`,
      }));
  }

  async start(
    videoEl: HTMLVideoElement,
    options: { deviceId?: string; preferLowRes?: boolean } = {}
  ): Promise<CameraResolution> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('مرورگر شما از دسترسی به دوربین پشتیبانی نمی‌کند.');
    }

    this.stop();
    this.videoEl = videoEl;

    const attempts: CameraResolution[] = options.preferLowRes
      ? [{ width: 640, height: 480, fps: 24 }]
      : [
          { width: 1280, height: 720, fps: 30 },
          { width: 640, height: 480, fps: 30 },
        ];

    let lastError: unknown = null;

    for (const target of attempts) {
      try {
        const videoConstraints: MediaTrackConstraints = {
          width: { ideal: target.width },
          height: { ideal: target.height },
          frameRate: { ideal: target.fps, max: 30 },
          ...(options.deviceId
            ? { deviceId: { exact: options.deviceId } }
            : { facingMode: 'user' }),
        };

        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: videoConstraints,
        });

        this.stream = stream;
        const track = stream.getVideoTracks()[0];
        if (track) {
          const settings = track.getSettings();
          this.activeDeviceId = settings.deviceId ?? options.deviceId ?? null;
          this.resolution = {
            width: settings.width ?? target.width,
            height: settings.height ?? target.height,
            fps: settings.frameRate ?? target.fps,
          };

          track.addEventListener('ended', () => {
            this.onCameraLost?.('جریان تصویر دوربین متوقف شد.');
          });
        }

        videoEl.srcObject = stream;
        videoEl.muted = true;
        videoEl.playsInline = true;
        await videoEl.play();
        return this.resolution;
      } catch (err) {
        lastError = err;
      }
    }

    throw lastError instanceof Error
      ? lastError
      : new Error('دسترسی به دوربین امکان‌پذیر نیست.');
  }

  async switchDevice(deviceId: string): Promise<CameraResolution> {
    if (!this.videoEl) {
      throw new Error('المان ویدیو آماده نیست.');
    }
    return this.start(this.videoEl, { deviceId });
  }

  stop(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
    }
  }
}

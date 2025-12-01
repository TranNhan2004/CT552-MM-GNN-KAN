import { Component, EventEmitter, Output } from '@angular/core';
import { FileLimit, UploadedFile } from '../../types/file';
import { Icon } from "../icon/icon";

@Component({
  selector: 'app-file-uploader',
  imports: [Icon],
  templateUrl: './file-uploader.html',
  styleUrl: './file-uploader.css'
})
export class FileUploader {
  @Output() filesChanged = new EventEmitter<UploadedFile[]>();

  uploadedFiles: UploadedFile[] = [];
  showModal: boolean = false;

  fileLimits: Record<UploadedFile['type'], FileLimit> = {
    video: {
      min: 0, max: 1, minDuration: 30, maxDuration: 120, required: false,
      accept: '.mp4'
    },
    image: {
      min: 3, max: 20, required: true,
      accept: '.jpg,.jpeg,.png'
    },
    text: {
      min: 0, max: 1, required: false,
      accept: '.txt'
    },
    audio: {
      min: 0, max: 1, minDuration: 30, maxDuration: 120, required: false,
      accept: '.mp3,.wav'
    }
  };

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  openModal() {
    this.showModal = true;
  }

  closeModal() {
    this.showModal = false;
  }

  async onFileSelected(event: Event, type: UploadedFile['type']) {
    const input = event.target as HTMLInputElement;
    if (!input.files) return;

    const files = Array.from(input.files);
    const existing = this.uploadedFiles.filter(f => f.type === type);
    const limit = this.fileLimits[type];
    const availableSlots = limit.max - existing.length;

    if (availableSlots <= 0) {
      alert(`Bạn chỉ có thể tải tối đa ${limit.max} file ${this.getTypeLabel(type)}.`);
      input.value = '';
      return;
    }

    const toAdd = files.slice(0, availableSlots);

    if (type === 'video' || type === 'audio') {
      const validatedFiles = await this.validateMediaDuration(toAdd, type);
      this.addFiles(validatedFiles, type);
    } else {
      this.addFiles(toAdd, type);
    }

    input.value = '';
  }

  private async validateMediaDuration(files: File[], type: 'video' | 'audio'): Promise<File[]> {
    const limit = this.fileLimits[type];
    const validFiles: File[] = [];

    for (const file of files) {
      const duration = await this.getMediaDuration(file, type);

      if (duration < (limit.minDuration || 0)) {
        alert(`File "${file.name}" quá ngắn. Tối thiểu ${limit.minDuration}s.`);
        continue;
      }

      if (duration > (limit.maxDuration || Infinity)) {
        alert(`File "${file.name}" quá dài. Tối đa ${limit.maxDuration}s.`);
        continue;
      }

      validFiles.push(file);
    }

    return validFiles;
  }

  private getMediaDuration(file: File, type: 'video' | 'audio'): Promise<number> {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const element = type === 'video'
        ? document.createElement('video')
        : document.createElement('audio');

      element.onloadedmetadata = () => {
        URL.revokeObjectURL(url);
        resolve(element.duration);
      };

      element.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(0);
      };

      element.src = url;
    });
  }

  private addFiles(files: File[], type: UploadedFile['type']) {
    const existing = this.uploadedFiles.filter(f => f.type === type);
    const newFiles = files.filter(
      file => !existing.some(f => f.file.name === file.name && f.file.size === file.size)
    );

    this.uploadedFiles = [
      ...this.uploadedFiles,
      ...newFiles.map(file => ({
        id: this.generateId(),
        type,
        file
      }))
    ];

    this.filesChanged.emit(this.uploadedFiles);
  }

  removeFile(id: string) {
    this.uploadedFiles = this.uploadedFiles.filter(f => f.id !== id);
    this.filesChanged.emit(this.uploadedFiles);
  }

  getTypeLabel(type: UploadedFile['type']): string {
    const labels = {
      video: 'Video',
      image: 'Hình ảnh',
      text: 'Văn bản',
      audio: 'Âm thanh'
    };
    return labels[type];
  }

  getFileCount(type: UploadedFile['type']): number {
    return this.uploadedFiles.filter(f => f.type === type).length;
  }

  getFileLimit(type: UploadedFile['type']): FileLimit {
    return this.fileLimits[type];
  }

  getTotalFileCount(): number {
    return this.uploadedFiles.length;
  }
}

import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AIService } from '../../services/ai';
import { HttpErrorResponse } from '@angular/common/http';
import { env } from '../../environments/env.dev';

interface AcceptedFile {
  id: string;
  type: 'video' | 'image' | 'text' | 'audio';
  file: File;
}


@Component({
  selector: 'app-home',
  imports: [FormsModule],
  templateUrl: './home.html',
  styleUrl: './home.css'
})
export class Home {
  uploadedFiles: AcceptedFile[] = [];
  text: string = "";
  result: string = "";

  private fileLimits: Record<AcceptedFile['type'], number> = {
    video: 5,
    image: 10,
    text: 10,
    audio: 10
  };

  constructor(private aiService: AIService) { }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  onFileSelected(event: Event, type: AcceptedFile['type']) {
    const input = event.target as HTMLInputElement;
    if (!input.files) return;

    const files = Array.from(input.files);
    const existing = this.uploadedFiles.filter(f => f.type === type);

    const availableSlots = this.fileLimits[type] - existing.length;
    if (availableSlots <= 0) {
      alert(`Bạn chỉ có thể tải tối đa ${this.fileLimits[type]} ${type}.`);
      input.value = '';
      return;
    }

    const toAdd = files.slice(0, availableSlots);

    const newFiles = toAdd.filter(
      file =>
        !existing.some(f => f.file.name === file.name && f.file.size === file.size)
    );

    this.uploadedFiles = [
      ...this.uploadedFiles,
      ...newFiles.map(file => ({
        id: this.generateId(),
        type,
        file
      }))
    ];

    input.value = '';
  }

  removeFile(id: string) {
    this.uploadedFiles = this.uploadedFiles.filter(f => f.id !== id);
  }

  onSubmit() {
    if (!this.text.trim() && this.uploadedFiles.length === 0) {
      alert('Vui lòng nhập văn bản hoặc tải lên ít nhất 1 file.');
      return;
    }

    console.log('Văn bản đã nhập', this.text);
    console.log('Files đã upload:', this.uploadedFiles);

    const files = this.uploadedFiles.map(item => item.file);
    this.aiService.predict(this.text, files).subscribe({
      next: (res) => {
        this.result = JSON.stringify(res);
      },
      error: (err: HttpErrorResponse) => {
        if (!env.production) {
          console.log(err);
        }
      }
    })
  }
}

import { HttpErrorResponse } from '@angular/common/http';
import { Component, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AIService } from '../../services/ai';
import { UploadedFile } from '../../types/file';
import { FileUploader } from '../file-uploader/file-uploader';
import { Icon } from '../icon/icon';

@Component({
  selector: 'app-input-section',
  imports: [FormsModule, FileUploader, Icon],
  templateUrl: './input-section.html',
  styleUrl: './input-section.css'
})
export class InputSection {
  text = signal<string>('');
  uploadedFiles = signal<UploadedFile[]>([]);
  isProcessing = signal<boolean>(false);
  isCompleted = signal<boolean>(false);
  errorMessage = signal<string>('');
  successMessage = signal<string>('');

  dataProcessed = output<number>();

  constructor(private aiService: AIService) {}

  onFilesChanged(files: UploadedFile[]) {
    this.uploadedFiles.set(files);
    this.errorMessage.set('');
    this.isCompleted.set(false);
  }

  onComplete() {
    const textValue = this.text();
    const files = this.uploadedFiles();

    // Validation
    if (!textValue.trim() && files.length === 0) {
      this.errorMessage.set('Vui lòng nhập văn bản hoặc tải lên ít nhất 1 file.');
      return;
    }

    const images = files.filter(f => f.type === 'image');
    const video = files.filter(f => f.type === 'video');

    if (images.length < 3 && video.length < 1) {
      this.errorMessage.set('Vui lòng tải lên ít nhất 3 hình ảnh hoặc 1 video (bắt buộc).');
      return;
    }

    // Upload and process
    this.errorMessage.set('');
    this.successMessage.set('');
    this.isProcessing.set(true);

    const fileObjects = files.map(item => item.file);

    this.aiService.upload(textValue, fileObjects).subscribe({
      next: (response) => {
        this.isProcessing.set(false);
        this.isCompleted.set(true);
        this.successMessage.set('Dữ liệu đã được tải lên thành công! Bây giờ bạn có thể dự đoán.');
        this.dataProcessed.emit(response.id);
      },
      error: (err: HttpErrorResponse) => {
        this.isProcessing.set(false);
        this.errorMessage.set('Đã xảy ra lỗi khi tải lên dữ liệu. Vui lòng thử lại.');
        console.error(err);
      }
    });
  }
}

import { HttpErrorResponse } from '@angular/common/http';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subject, takeUntil } from 'rxjs';
import { FileUploader } from "../../components/file-uploader/file-uploader";
import { Icon } from "../../components/icon/icon";
import { ResultDisplay } from "../../components/result-display/result-display";
import { AIService } from '../../services/ai';
import { ModelService } from '../../services/model';
import { UploadedFile } from '../../types/file';
import { ModelType } from '../../types/model';
import { ResultRes } from '../../types/result';

const MOCK_RESULT: ResultRes = {
  id: 1,
  text: "",
  processedTexts: [],
  audioUrls: [],
  imageUrls: [],
  labelIdx: 0,
  labelName: "abc",
  prob: 0.5,
  weights: []
};


@Component({
  selector: 'app-home',
  imports: [FormsModule, FileUploader, ResultDisplay, Icon],
  templateUrl: './home.html',
  styleUrl: './home.css'
})
export class Home implements OnInit, OnDestroy {
  text: string = '';
  uploadedFiles: UploadedFile[] = [];
  result: ResultRes | null = MOCK_RESULT;
  isLoading: boolean = false;
  errorMessage: string = '';
  selectedModel: ModelType = 'mobilenetv3small';

  private destroy$ = new Subject<void>();

  constructor(
    private aiService: AIService,
    private modelService: ModelService
  ) {}

  ngOnInit() {
    this.modelService.selectedModel$
      .pipe(takeUntil(this.destroy$))
      .subscribe(model => {
        this.selectedModel = model;
      });
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }

  onFilesChanged(files: UploadedFile[]) {
    this.uploadedFiles = files;
    this.errorMessage = '';
  }

  onSubmit() {
    if (!this.text.trim() && this.uploadedFiles.length === 0) {
      this.errorMessage = 'Vui lòng nhập văn bản hoặc tải lên ít nhất 1 file.';
      return;
    }

    const images = this.uploadedFiles.filter(f => f.type === 'image');
    if (images.length < 3) {
      this.errorMessage = 'Vui lòng tải lên ít nhất 3 hình ảnh (bắt buộc).';
      return;
    }

    this.errorMessage = '';
    this.isLoading = true;

    const files = this.uploadedFiles.map(item => item.file);

    this.aiService.predict(this.selectedModel, this.text, files).subscribe({
      next: (res) => {
        this.result = res;
        this.isLoading = false;
      },
      error: (err: HttpErrorResponse) => {
        this.errorMessage = 'Đã xảy ra lỗi khi dự đoán. Vui lòng thử lại.';
        this.isLoading = false;
        console.error(err);
      }
    });
  }
}

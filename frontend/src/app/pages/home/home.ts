import { HttpErrorResponse } from '@angular/common/http';
import { ChangeDetectorRef, Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subject, takeUntil } from 'rxjs';
import { FileUploader } from "../../components/file-uploader/file-uploader";
import { GraphDisplay } from "../../components/graph-display/graph-display";
import { Icon } from "../../components/icon/icon";
import { NodeList } from "../../components/node-list/node-list";
import { ResultDisplay } from "../../components/result-display/result-display";
import { AIService } from '../../services/ai';
import { ModelService } from '../../services/model';
import { UploadedFile } from '../../types/file';
import { ModelType } from '../../types/model';
import { SelectedNodesData } from '../../types/node-list';
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
  imports: [FormsModule, FileUploader, ResultDisplay, Icon, GraphDisplay, NodeList],
  templateUrl: './home.html',
  styleUrl: './home.css'
})
export class Home implements OnInit, OnDestroy {
  text: string = '';
  uploadedFiles: UploadedFile[] = [];
  result: ResultRes | null = null;
  isLoading: boolean = false;
  errorMessage: string = '';
  selectedModel: ModelType = 'mobilenetv3large';
  selectionData: SelectedNodesData | null = null;
  resId: number | null = null;

  private destroy$ = new Subject<void>();

  constructor(
    private aiService: AIService,
    private modelService: ModelService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit() {
    // this.resId = Number(localStorage.getItem("RES_ID"));
    // if (this.resId) {
    //   this.aiService.get(this.resId).subscribe({
    //     next: (res) => {
    //       this.result = res;
    //     }
    //   });
    // }

    this.aiService.get(1).subscribe({
      next: (res) => {
        Promise.resolve().then(() => {
          this.result = res;
        });
      }
    });

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

  onSelectionChanged(data: SelectedNodesData) {
    this.selectionData = { ...data };
    this.cdr.detectChanges();
  }


  onSubmit() {
    if (!this.text.trim() && this.uploadedFiles.length === 0) {
      this.errorMessage = 'Vui lòng nhập văn bản hoặc tải lên ít nhất 1 file.';
      return;
    }

    const images = this.uploadedFiles.filter(f => f.type === 'image');
    const video = this.uploadedFiles.filter(f => f.type === 'video')
    if (images.length < 3 && video.length < 1) {
      this.errorMessage = 'Vui lòng tải lên ít nhất 3 hình ảnh hoặc 1 video (bắt buộc).';
      return;
    }

    this.errorMessage = '';
    this.isLoading = true;

    const files = this.uploadedFiles.map(item => item.file);

    this.aiService.predict(this.selectedModel, this.text, files).subscribe({
      next: (res) => {
        this.result = res;
        this.isLoading = false;
        localStorage.setItem("RES_ID", this.result.id.toString());
      },
      error: (err: HttpErrorResponse) => {
        this.errorMessage = 'Đã xảy ra lỗi khi dự đoán. Vui lòng thử lại.';
        this.isLoading = false;
        console.error(err);
      }
    });
  }
}

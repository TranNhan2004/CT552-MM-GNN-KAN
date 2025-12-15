import { HttpErrorResponse } from '@angular/common/http';
import { Component, computed, input, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AIService } from '../../services/ai';
import { ModelType, SelectModelOptions } from '../../types/model';
import { SelectedNodesData } from '../../types/node-list';
import { PredictionResult, PredictionType, ResultRes } from '../../types/result';
import { GraphDisplay } from '../graph-display/graph-display';
import { Icon } from '../icon/icon';

@Component({
  selector: 'app-prediction-column',
  imports: [FormsModule, Icon, GraphDisplay],
  templateUrl: './prediction-column.html',
  styleUrl: './prediction-column.css'
})
export class PredictionColumn implements OnInit{
  predictionType = input.required<PredictionType>();
  selectModelOptions = input.required<SelectModelOptions[]>();
  dataId = input<number | null>(null);

  selectedModel = signal<ModelType>('mobilenetv3small');
  isLoading = signal(false);
  errorMessage = signal('');
  result = signal<ResultRes | null>(null);
  selectionData = signal<SelectedNodesData | null>(null);
  openDropdown = signal<boolean>(false);

  showGraph = computed(() => {
    const type = this.predictionType();
    return type === 'img-txt' || type === 'full';
  });

  predictionResult = computed((): PredictionResult | null => {
    const result = this.result();
    if (!result) return null;

    const type = this.predictionType();
    switch (type) {
      case 'img':
        return {
          labelName: result.imgLabelName,
          prob: result.imgProb
        };
      case 'img-txt':
        return {
          labelName: result.imgTxtLabelName,
          prob: result.imgTxtProb,
          weights: result.imgTxtWeights
        };
      case 'full':
        return {
          labelName: result.fullLabelName,
          prob: result.fullProb,
          weights: result.fullWeights
        };
      default:
        return null;
    }
  });

  notUseAudioLabels = new Set<string>([
    'Chợ nổi Cái Răng',
    'Đua bò Bảy Núi',
    'Nghề đan tre',
    'Nghề dệt chiếu'
  ]);

  constructor(private aiService: AIService) {}

  ngOnInit(): void {
    // this.aiService.get(15).subscribe({
    //   next: (data) => {
    //     this.result.set(data);
    //     console.log('Sample data for graph display:', data);
    //   },
    //   error: (err: HttpErrorResponse) => {
    //     console.error('Error fetching sample data:', err);
    //   }
    // });
  }

  select(opt: any) {
    this.selectedModel.set(opt.value);
    this.openDropdown.set(false);
  }

  selectedModelObj() {
    return this.selectModelOptions().find(o => o.value === this.selectedModel());
  }

  getTitle(): string {
    const type = this.predictionType();
    switch (type) {
      case 'img': return 'Hình ảnh';
      case 'img-txt': return 'Hình ảnh & Văn bản';
      case 'full': return 'Hình ảnh & Văn bản & Âm thanh';
      default: return '';
    }
  }

  onPredict() {
    const id = this.dataId();
    if (!id) {
      this.errorMessage.set('Vui lòng hoàn tất upload dữ liệu trước.');
      return;
    }

    this.errorMessage.set('');
    this.isLoading.set(true);

    const model = this.selectedModel();
    const type = this.predictionType();

    let request;
    switch (type) {
      case 'img':
        request = this.aiService.predictCNN(id, model);
        break;
      case 'img-txt':
        request = this.aiService.predictImageText(id, model);
        break;
      case 'full':
        request = this.aiService.predictFull(id, model);
        break;
    }

    request.subscribe({
      next: (result) => {
        if (this.notUseAudioLabels.has(result.fullLabelName)) {
          result.audioUrls = [];
        }

        this.result.set(result);
        this.isLoading.set(false);
      },
      error: (err: HttpErrorResponse) => {
        this.errorMessage.set('Đã xảy ra lỗi khi dự đoán. Vui lòng thử lại.');
        this.isLoading.set(false);
        console.error(err);
      }
    });
  }

  onSelectionChanged(data: SelectedNodesData) {
    this.selectionData.set(data);
  }

  getConfidenceColor(prob: number): string {
    if (prob >= 0.8) return 'text-green-600';
    if (prob >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  }

  getConfidenceBackground(prob: number): string {
    if (prob >= 0.8) return 'bg-green-100 border-green-300';
    if (prob >= 0.6) return 'bg-yellow-100 border-yellow-300';
    return 'bg-red-100 border-red-300';
  }

  getProgressBarColor(prob: number): string {
    if (prob >= 0.8) return 'bg-gradient-to-r from-green-500 to-emerald-600';
    if (prob >= 0.6) return 'bg-gradient-to-r from-yellow-500 to-orange-600';
    return 'bg-gradient-to-r from-red-500 to-rose-600';
  }
}

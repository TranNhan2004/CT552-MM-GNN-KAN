import { Component, Input } from '@angular/core';
import { ResultRes } from '../../types/result';
import { Icon } from "../icon/icon";

@Component({
  selector: 'app-result-display',
  imports: [Icon],
  templateUrl: './result-display.html',
  styleUrl: './result-display.css'
})
export class ResultDisplay {
  @Input() result: ResultRes | null = null;
  @Input() isLoading: boolean = false;

  getConfidenceLevel(prob: number): string {
    if (prob >= 0.8) return 'Cao';
    if (prob >= 0.6) return 'Trung bình';
    return 'Thấp';
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

  getConfidenceGradient(prob: number): string {
    if (prob >= 0.8) return 'from-green-500 to-emerald-600';
    if (prob >= 0.6) return 'from-yellow-500 to-orange-600';
    return 'from-red-500 to-rose-600';
  }

  getProgressBarColor(prob: number): string {
    if (prob >= 0.8) return 'bg-gradient-to-r from-green-500 to-emerald-600';
    if (prob >= 0.6) return 'bg-gradient-to-r from-yellow-500 to-orange-600';
    return 'bg-gradient-to-r from-red-500 to-rose-600';
  }
}


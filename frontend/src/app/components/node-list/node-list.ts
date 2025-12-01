import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';
import { env } from '../../environments/env.dev';
import { EdgeThreshold, SelectedNodesData } from '../../types/node-list';
import { ResultRes } from '../../types/result';
import { Icon } from "../icon/icon";

@Component({
  selector: 'app-node-list',
  imports: [Icon],
  templateUrl: './node-list.html',
  styleUrl: './node-list.css'
})
export class NodeList implements OnChanges {
  @Input() result: ResultRes | null = null;
  @Output() selectionChanged = new EventEmitter<SelectedNodesData>();

  showModal = false;
  selectedImageIndices: Set<number> = new Set();
  selectedTextIndices: Set<number> = new Set();
  selectedAudioIndices: Set<number> = new Set();
  selectedThreshold: EdgeThreshold = 80;

  thresholds: EdgeThreshold[] = [80, 60, 40, 20];

  get env() {
    return env;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['result'] && this.result) {
      this.initializeSelection();
    }
  }

  private initializeSelection() {
    if (!this.result) return;

    // Auto-select all nodes initially
    this.selectedImageIndices.clear();
    this.selectedTextIndices.clear();
    this.selectedAudioIndices.clear();

    this.emitSelection();
  }

  openModal() {
    this.showModal = true;
  }

  closeModal() {
    this.showModal = false;
  }

  toggleImage(index: number) {
    if (this.selectedImageIndices.has(index)) {
      this.selectedImageIndices.delete(index);
    } else {
      this.selectedImageIndices.add(index);
    }
    this.emitSelection();
  }

  toggleText(index: number) {
    if (this.selectedTextIndices.has(index)) {
      this.selectedTextIndices.delete(index);
    } else {
      this.selectedTextIndices.add(index);
    }
    this.emitSelection();
  }

  toggleAudio(index: number) {
    if (this.selectedAudioIndices.has(index)) {
      this.selectedAudioIndices.delete(index);
    } else {
      this.selectedAudioIndices.add(index);
    }
    this.emitSelection();
  }

  isImageSelected(index: number): boolean {
    return this.selectedImageIndices.has(index);
  }

  isTextSelected(index: number): boolean {
    return this.selectedTextIndices.has(index);
  }

  isAudioSelected(index: number): boolean {
    return this.selectedAudioIndices.has(index);
  }

  selectAllImages() {
    if (!this.result) return;
    (this.result.imageUrls || []).forEach((_, idx) => this.selectedImageIndices.add(idx));
    this.emitSelection();
  }

  deselectAllImages() {
    this.selectedImageIndices.clear();
    this.emitSelection();
  }

  selectAllTexts() {
    if (!this.result) return;
    (this.result.processedTexts || []).forEach((_, idx) => this.selectedTextIndices.add(idx));
    this.emitSelection();
  }

  deselectAllTexts() {
    this.selectedTextIndices.clear();
    this.emitSelection();
  }

  selectAllAudios() {
    if (!this.result) return;
    (this.result.audioUrls || []).forEach((_, idx) => this.selectedAudioIndices.add(idx));
    this.emitSelection();
  }

  deselectAllAudios() {
    this.selectedAudioIndices.clear();
    this.emitSelection();
  }

  selectAll() {
    this.selectAllImages();
    this.selectAllTexts();
    this.selectAllAudios();
  }

  deselectAll() {
    this.selectedImageIndices.clear();
    this.selectedTextIndices.clear();
    this.selectedAudioIndices.clear();
    this.emitSelection();
  }

  setThreshold(threshold: EdgeThreshold) {
    this.selectedThreshold = threshold;
    this.emitSelection();
  }

  private emitSelection() {
    this.selectionChanged.emit({
      imageIndices: [...this.selectedImageIndices],
      textIndices: [...this.selectedTextIndices],
      audioIndices: [...this.selectedAudioIndices],
      threshold: this.selectedThreshold
    });
  }

  getSelectedCount(): number {
    return this.selectedImageIndices.size +
           this.selectedTextIndices.size +
           this.selectedAudioIndices.size;
  }

  getTotalCount(): number {
    if (!this.result) return 0;
    return (this.result.imageUrls?.length || 0) +
           (this.result.processedTexts?.length || 0) +
           (this.result.audioUrls?.length || 0);
  }

  getImageCount(): number {
    return this.result?.imageUrls?.length || 0;
  }

  getTextCount(): number {
    return this.result?.processedTexts?.length || 0;
  }

  getAudioCount(): number {
    return this.result?.audioUrls?.length || 0;
  }
}

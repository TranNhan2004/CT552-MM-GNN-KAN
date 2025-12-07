import { Component, OnChanges, SimpleChanges, input, output, signal } from '@angular/core';
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
  result = input<ResultRes | null>(null);
  missingAudio = input<boolean>(false);

  selectionChanged = output<SelectedNodesData>();

  showModal = signal(false);
  selectedImageIndices = signal<Set<number>>(new Set());
  selectedTextIndices = signal<Set<number>>(new Set());
  selectedAudioIndices = signal<Set<number>>(new Set());
  selectedThreshold = signal<EdgeThreshold>(100);

  thresholds: EdgeThreshold[] = [100, 80, 60, 40, 20];

  get env() {
    return env;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['result'] && this.result()) {
      this.initializeSelection();
    }
  }

  private initializeSelection() {
    if (!this.result()) return;

    // Auto-select all nodes initially
    this.selectedImageIndices.set(new Set());
    this.selectedTextIndices.set(new Set());
    this.selectedAudioIndices.set(new Set());

    this.emitSelection();
  }

  openModal() {
    this.showModal.set(true);
  }

  closeModal() {
    this.showModal.set(false);
  }

  toggleImage(index: number) {
    this.selectedImageIndices.update(set => {
      const newSet = new Set(set);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
    this.emitSelection();
  }

  toggleText(index: number) {
    this.selectedTextIndices.update(set => {
      const newSet = new Set(set);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
    this.emitSelection();
  }

  toggleAudio(index: number) {
    this.selectedAudioIndices.update(set => {
      const newSet = new Set(set);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
    this.emitSelection();
  }

  isImageSelected(index: number): boolean {
    return this.selectedImageIndices().has(index);
  }

  isTextSelected(index: number): boolean {
    return this.selectedTextIndices().has(index);
  }

  isAudioSelected(index: number): boolean {
    return this.selectedAudioIndices().has(index);
  }

  selectAllImages() {
    if (!this.result()) return;
    const newSet = new Set<number>();
    (this.result()!.imageUrls || []).forEach((_, idx) => newSet.add(idx));
    this.selectedImageIndices.set(newSet);
    this.emitSelection();
  }

  deselectAllImages() {
    this.selectedImageIndices.set(new Set());
    this.emitSelection();
  }

  selectAllTexts() {
    if (!this.result()) return;
    const newSet = new Set<number>();
    (this.result()!.processedTexts || []).forEach((_, idx) => newSet.add(idx));
    this.selectedTextIndices.set(newSet);
    this.emitSelection();
  }

  deselectAllTexts() {
    this.selectedTextIndices.set(new Set());
    this.emitSelection();
  }

  selectAllAudios() {
    if (!this.result()) return;
    const newSet = new Set<number>();
    (this.result()!.audioUrls || []).forEach((_, idx) => newSet.add(idx));
    this.selectedAudioIndices.set(newSet);
    this.emitSelection();
  }

  deselectAllAudios() {
    this.selectedAudioIndices.set(new Set());
    this.emitSelection();
  }

  selectAll() {
    this.selectAllImages();
    this.selectAllTexts();
    if (!this.missingAudio()) {
      this.selectAllAudios();
    }
  }

  deselectAll() {
    this.selectedImageIndices.set(new Set());
    this.selectedTextIndices.set(new Set());
    this.selectedAudioIndices.set(new Set());
    this.emitSelection();
  }

  setThreshold(threshold: EdgeThreshold) {
    this.selectedThreshold.set(threshold);
    this.emitSelection();
  }

  private emitSelection() {
    this.selectionChanged.emit({
      imageIndices: [...this.selectedImageIndices()],
      textIndices: [...this.selectedTextIndices()],
      audioIndices: [...this.selectedAudioIndices()],
      threshold: this.selectedThreshold()
    });
  }

  getSelectedCount(): number {
    let count = this.selectedImageIndices().size + this.selectedTextIndices().size;
    if (!this.missingAudio()) {
      count += this.selectedAudioIndices().size;
    }
    return count;
  }

  getTotalCount(): number {
    if (!this.result()) return 0;
    let count = (this.result()!.imageUrls?.length || 0) +
                (this.result()!.processedTexts?.length || 0);
    if (!this.missingAudio()) {
      count += (this.result()!.audioUrls?.length || 0);
    }
    return count;
  }

  getImageCount(): number {
    return this.result()?.imageUrls?.length || 0;
  }

  getTextCount(): number {
    return this.result()?.processedTexts?.length || 0;
  }

  getAudioCount(): number {
    return this.result()?.audioUrls?.length || 0;
  }
}

import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { ModelType } from '../types/model';

@Injectable({
  providedIn: 'root'
})
export class ModelService {
  private selectedModelSubject = new BehaviorSubject<ModelType>('mobilenetv3small');
  public selectedModel$: Observable<ModelType> = this.selectedModelSubject.asObservable();

  constructor() {}

  getSelectedModel(): ModelType {
    return this.selectedModelSubject.value;
  }

  setSelectedModel(model: ModelType): void {
    this.selectedModelSubject.next(model);
  }
}

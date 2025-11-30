import { Component, ElementRef, HostListener, OnInit } from '@angular/core';
import { ModelService } from '../../services/model';
import { ModelType } from '../../types/model';
import { Icon } from "../icon/icon";

@Component({
  selector: 'app-navbar',
  imports: [Icon],
  templateUrl: './navbar.html',
  styleUrls: ['./navbar.css']
})
export class Navbar implements OnInit {
  dropdownOpen = false;
  selectedModel: ModelType = 'mobilenetv3small';

  models: Record<string, ModelType> = {
    "ResNet": "resnet",
    "RegNet": "regnet",
    "MobileNet V3 Small": "mobilenetv3small",
    "MobileNet V3 Large": "mobilenetv3large",
    "DenseNet": "densenet",
    "ShuffleNet": "shufflenet"
  };

  modelLabels = Object.keys(this.models);

  constructor(
    private eRef: ElementRef,
    private modelService: ModelService
  ) {}

  ngOnInit() {
    this.modelService.selectedModel$.subscribe(model => {
      this.selectedModel = model;
    });
  }

  toggleDropdown() {
    this.dropdownOpen = !this.dropdownOpen;
  }

  selectModel(modelLabel: string) {
    const modelValue = this.models[modelLabel];
    this.modelService.setSelectedModel(modelValue);
    this.dropdownOpen = false;
  }

  getSelectedModelLabel(): string {
    const entry = Object.entries(this.models).find(([_, value]) => value === this.selectedModel);
    return entry ? entry[0] : 'MobileNet V3 Small';
  }

  @HostListener('document:click', ['$event'])
  clickOutside(event: Event) {
    if (this.dropdownOpen && !this.eRef.nativeElement.contains(event.target)) {
      this.dropdownOpen = false;
    }
  }
}

import { Component, signal } from '@angular/core';
import { InputSection } from "../../components/input-section/input-section";
import { PredictionColumn } from "../../components/prediction-column/prediction-column";
import { SelectModelOptions } from '../../types/model';

@Component({
  selector: 'app-home',
  imports: [InputSection, PredictionColumn],
  templateUrl: './home.html',
  styleUrl: './home.css'
})
export class Home {
  uploadedDataId = signal<number | null>(null);
  imageSelectModelOptions: SelectModelOptions[] = [
    { label: "ResNet18", value: "resnet", best: false, },
    { label: "RegNetX400MF", value: "regnet", best: false },
    { label: "MobileNetV3 Large", value: "mobilenetv3large", best: false },
    { label: "MobileNetV3 Small", value: "mobilenetv3small", best: false },
    { label: "DenseNet121", value: "densenet", best: true },
    { label: "ShuffleNetV2 x1.0", value: "shufflenet", best: false }
  ];

  imageTextSelectModelOptions: SelectModelOptions[] = [
    { label: "ResNet18 + MGAT + FastKAN", value: "resnet", best: false, },
    { label: "RegNetX400MF + MGAT + MLP", value: "regnet", best: false },
    { label: "MobileNetV3 Large + MGAT + FastKAN", value: "mobilenetv3large", best: true },
    { label: "MobileNetV3 Small + MGAT + MLP", value: "mobilenetv3small", best: false },
    { label: "DenseNet121 + MGAT + FastKAN", value: "densenet", best: false },
    { label: "ShuffleNetV2 x1.0 + MGAT + FastKAN", value: "shufflenet", best: false }
  ];

  fullSelectModelOptions: SelectModelOptions[] = [
    { label: "ResNet18 + MGAT + FastKAN", value: "resnet", best: false, },
    { label: "RegNetX400MF + MGAT + MLP", value: "regnet", best: false },
    { label: "MobileNetV3 Large + MGAT + FastKAN", value: "mobilenetv3large", best: false },
    { label: "MobileNetV3 Small + MGAT + MLP", value: "mobilenetv3small", best: false },
    { label: "DenseNet121 + MGAT + FastKAN", value: "densenet", best: true },
    { label: "ShuffleNetV2 x1.0 + MGAT + MLP", value: "shufflenet", best: false }
  ];

  onDataProcessed(id: number) {
    this.uploadedDataId.set(id);
    console.log('Data uploaded with ID:', id);
  }

  onRefresh() {
    window.location.reload();
  }
}

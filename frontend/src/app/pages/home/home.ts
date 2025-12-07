import { Component, signal } from '@angular/core';
import { InputSection } from "../../components/input-section/input-section";
import { PredictionColumn } from "../../components/prediction-column/prediction-column";

@Component({
  selector: 'app-home',
  imports: [InputSection, PredictionColumn],
  templateUrl: './home.html',
  styleUrl: './home.css'
})
export class Home {
  uploadedDataId = signal<number | null>(null);

  onDataProcessed(id: number) {
    this.uploadedDataId.set(id);
    console.log('Data uploaded with ID:', id);
  }
}

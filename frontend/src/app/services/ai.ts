import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { env } from '../environments/env.dev';

@Injectable({
  providedIn: 'root'
})
export class AIService {
  constructor(private httpClient: HttpClient) { }

  predict(text: string, uploadedFiles: File[]) {
    const formData = new FormData();
    formData.append('text', text);

    uploadedFiles.forEach(file => {
      formData.append('files', file, file.name);
    });

    return this.httpClient.post(`${env.apiUrl}/ai/predict`, formData);
  }
}


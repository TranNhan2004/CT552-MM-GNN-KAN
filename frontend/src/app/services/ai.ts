import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map } from 'rxjs';
import { env } from '../environments/env.dev';
import { ResultRes } from '../types/result';
import { keysToCamel } from '../utils/case';

@Injectable({
  providedIn: 'root'
})
export class AIService {
  constructor(private httpClient: HttpClient) { }

  predict(modelName: string, text: string, uploadedFiles: File[]) {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('model_name', modelName);

    uploadedFiles.forEach(file => {
      formData.append('files', file, file.name);
    });

    return this.httpClient.post(`${env.apiUrl}/predict`, formData).pipe(
      map((res: any) => keysToCamel(res) as ResultRes)
    );
  }

  get(id: number) {
    return this.httpClient.get(`${env.apiUrl}/result/${id}`).pipe(
      map((res: any) => keysToCamel(res) as ResultRes)
    );;
  }

}


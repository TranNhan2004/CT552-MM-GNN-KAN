import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map } from 'rxjs';
import { env } from '../environments/env.dev';
import { ResultRes, UploadRes } from '../types/result';
import { keysToCamel } from '../utils/case';

import { ModelType } from '../types/model';



@Injectable({
  providedIn: 'root'
})
export class AIService {
  constructor(private httpClient: HttpClient) { }

  upload(text: string, uploadedFiles: File[]) {
    const formData = new FormData();
    formData.append('text', text);

    uploadedFiles.forEach(file => {
      formData.append('files', file, file.name);
    });

    return this.httpClient.post(`${env.apiUrl}/api/upload`, formData).pipe(
      map((res: any) => {
        console.log(res);
        return keysToCamel(res) as UploadRes;
      })
    );
  }

  predictCNN(dataId: number, model: ModelType) {
    return this.httpClient.post<ResultRes>(
      `${env.apiUrl}/api/predict/cnn`,
      { id: dataId, model }
    ).pipe(
      map((res: any) => keysToCamel(JSON.parse(res)) as ResultRes)
    );
  }

  predictImageText(dataId: number, model: ModelType) {
    return this.httpClient.post<ResultRes>(
      `${env.apiUrl}/api/predict/img-txt`,
      { id: dataId, model }
    ).pipe(
      map((res: any) => keysToCamel(JSON.parse(res)) as ResultRes)
    );
  }

  predictFull(dataId: number, model: ModelType) {
    return this.httpClient.post<ResultRes>(
      `${env.apiUrl}/api/predict/full`,
      { id: dataId, model }
    ).pipe(
      map((res: any) => keysToCamel(JSON.parse(res)) as ResultRes)
    );
  }

  get(id: number) {
    return this.httpClient.get(`${env.apiUrl}/api/results/${id}`).pipe(
      map((res: any) => {
        return keysToCamel(JSON.parse(res)) as ResultRes;
      })
    );
  }

}


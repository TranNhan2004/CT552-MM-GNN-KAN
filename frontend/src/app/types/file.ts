export interface UploadedFile {
  id: string;
  type: 'video' | 'image' | 'text' | 'audio';
  file: File;
  duration?: number;
}

export interface FileLimit {
  min: number;
  max: number;
  minDuration?: number;
  maxDuration?: number;
  required: boolean;
  accept: string;
}

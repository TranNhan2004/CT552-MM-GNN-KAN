export type ModelType =
  | 'resnet'
  | 'regnet'
  | 'mobilenetv3small'
  | 'mobilenetv3large'
  | 'densenet'
  | 'shufflenet';

export interface ModelInfo {
  name: string;
  displayName: string;
  description?: string;
}

export interface SelectModelOptions {
  value: ModelType;
  label: string;
  best: boolean;
}

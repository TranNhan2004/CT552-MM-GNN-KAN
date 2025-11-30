interface Word {
  word: string;
  sentence: string;
}

export interface ResultRes {
  id: number;
  text: string;
  processedTexts: Word[];
  imageUrls: string[];
  audioUrls: string[];
  labelIdx: number;
  labelName: string;
  prob: number;
  weights: number[][];
}

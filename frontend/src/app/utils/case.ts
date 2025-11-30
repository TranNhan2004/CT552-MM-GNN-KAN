function camelToSnake(str: string): string {
  return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
}

function snakeToCamel(str: string): string {
  return str.replace(/(_\w)/g, m => m[1].toUpperCase());
}

export function keysToSnake(obj: any): any {
  if (Array.isArray(obj)) return obj.map(keysToSnake);
  if (obj && obj.constructor === Object) {
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [camelToSnake(k), keysToSnake(v)])
    );
  }
  return obj;
}

export function keysToCamel(obj: any): any {
  if (Array.isArray(obj)) return obj.map(keysToCamel);
  if (obj && obj.constructor === Object) {
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [snakeToCamel(k), keysToCamel(v)])
    );
  }
  return obj;
}


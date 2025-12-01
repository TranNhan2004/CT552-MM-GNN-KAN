const isObject = (o: any) => {
  return o === Object(o) && !Array.isArray(o) && typeof o !== 'function';
};

function snakeToCamel(str: string): string {
  return str.replace(/([-_][a-z])/ig, ($1) => {
    return $1.toUpperCase()
      .replace('-', '')
      .replace('_', '');
  });
}

export function keysToCamel(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map((i) => keysToCamel(i));
  }

  if (isObject(obj)) {
    const n: any = {};
    Object.keys(obj).forEach((k) => {
      n[snakeToCamel(k)] = keysToCamel(obj[k]);
    });
    return n;
  }

  return obj;
}

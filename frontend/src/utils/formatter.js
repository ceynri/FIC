export const BTYE_PER_KB = 2 ** 10;
export const BTYE_PER_MB = 2 ** 20;

/**
 * 将数值转为百分数表示
 */
export function percentFormat(number, remainder = 2) {
  return `${(number * 100).toFixed(remainder)}%`;
}

/**
 * size大小字符串格式化
 */
export function sizeFormat(byte, remainder = 2) {
  if (byte < BTYE_PER_KB) {
    return `${byte} B`;
  }
  if (byte < BTYE_PER_MB) {
    return `${(byte / BTYE_PER_KB).toFixed(remainder)} KB`;
  }
  return `${(byte / BTYE_PER_MB).toFixed(remainder)} MB`;
}

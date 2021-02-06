export default function getKebabCase(str) {
  const arr = str.split('');
  const newStr = arr.map((item) => {
    // 使用toUpperCase()方法检测当前字符是否为大写
    const lowerCase = item.toLowerCase();
    if (item === lowerCase) {
      return item;
    }
    return `-${lowerCase}`;
  });
  return newStr.join('');
}

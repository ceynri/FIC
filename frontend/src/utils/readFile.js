export default async function(file, type = 'dataUrl') {
  return new Promise((resolve, reject) => {
    const fileReader = new FileReader();
    fileReader.addEventListener('load', () => {
      resolve(fileReader.result);
    });
    switch (type) {
      case 'base64':
      case 'dataUrl':
        fileReader.readAsDataURL(file);
        break;
      case 'blob':
      case 'btye':
      case 'arrayBuffer':
        fileReader.readAsArrayBuffer(file);
        break;
      default:
        reject(new Error(`错误的type类型：${type}`));
    }
  });
}

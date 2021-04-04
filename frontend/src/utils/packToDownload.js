import axios from 'axios';
import JSZip from 'jszip';
import fileSaver from 'file-saver';

const getFile = async (url) => {
  const data = await axios({
    method: 'get',
    url,
    responseType: 'arraybuffer',
  });
  return data.data;
};

/**
 * 下载文件打包为zip文件并保存到本地
 * @param {Object} itemList like { data, name }
 */
export default async function packToDownload(itemList) {
  if (!itemList.length > 0) return;
  const zip = new JSZip();
  const cache = {};
  const promises = [];
  itemList.forEach((file) => {
    const promise = getFile(file.data).then((data) => {
      // 下载文件, 并存成ArrayBuffer对象
      zip.file(file.name, data, { binary: true }); // 逐个添加文件
      cache[file.name] = data;
    });
    promises.push(promise);
  });

  await Promise.all(promises);
  // 生成二进制流
  const result = await zip.generateAsync({ type: 'blob' });
  // 利用file-saver保存文件
  fileSaver.saveAs(result, `fic-${Date.now()}.zip`);
}

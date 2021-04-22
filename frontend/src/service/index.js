import axios from 'axios';

export async function demoProcess(file, options) {
  // 文件包装
  const data = new FormData();
  data.append('file', file);
  data.append('feature_model', options.featureModel);
  data.append('quality_level', options.qualityLevel);

  // API请求
  const res = await axios.post('/api/demo_process', data, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  // 获取返回结果
  return res.data;
}

export async function compress(files) {
  const data = new FormData();
  files.forEach((file) => {
    data.append('files', file);
  });
  const res = await axios.post('/api/compress', data, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
}

export async function decompress(files) {
  const data = new FormData();
  files.forEach((file) => {
    data.append('files', file);
  });
  const res = await axios.post('/api/decompress', data, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
}

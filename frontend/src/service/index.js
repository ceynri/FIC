import axios from 'axios';

// const baseUrl = 'http://127.0.0.1/api';
// TODO 检查该ip
const baseUrl = 'http://172.30.45.58/api';

// eslint-disable-next-line import/prefer-default-export
export function uploads(file) {
  const data = new FormData();
  data.append('file', file);
  return axios.post(`${baseUrl}/uploads`, data, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
}

export async function demoProcess(file) {
  const data = new FormData();
  data.append('file', file);
  const res = await axios.post(`${baseUrl}/demo_process`, data, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
}

export async function compress(files) {
  const data = new FormData();
  files.forEach((file) => {
    data.append('files', file);
  });
  const res = await axios.post(`${baseUrl}/compress`, data, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
}

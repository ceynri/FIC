import axios from 'axios';

const baseUrl = 'http://fic.ceynri.cn/api';

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

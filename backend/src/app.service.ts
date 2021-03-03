import { Injectable } from '@nestjs/common';

import * as child_process from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

import * as moment from 'moment';

import recursiveMkdir from './utils/recursiveMkdir';

@Injectable()
export class AppService {
  readonly basePath = '/data/fic';

  getHello(): string {
    return 'Hello World!';
  }

  getPath(file): string {
    const date = moment().format('YYYY/MM/DD');
    return path.join(this.basePath, './uploads', date, file.originalname);
  }

  async uploadFile(file, storagePath = '') {
    if (!storagePath) {
      storagePath = this.getPath(file);
    }
    await recursiveMkdir(storagePath);

    console.debug(file);
    const writeStream = fs.createWriteStream(storagePath);
    writeStream.write(file.buffer);
    return new Promise((resolve) => {
      writeStream.on('drain', () => {
        console.debug('done');
        resolve(true);
      });
    });
  }

  async demoProcess(file) {
    const path = this.getPath(file);
    await this.uploadFile(file, path);
    // TODO
    return new Promise((resolve, reject) => {
      const workerProcess = child_process.exec(
        `python ./service/demo.py ${path}`,
        (error, stdout, stderr) => {
          if (error) {
            console.log(error.stack);
            console.log(`Error code ${error.code}`);
            console.log(`Signal received ${error.signal}`);
            reject(error);
          }
          console.log(`stdout: ${stdout}`);
          console.log(`stderr: ${stderr}`);

          const fullName = file.originalname;
          const name = fullName.slice(0, fullName.lastIndexOf('.'));
          const ext = fullName.slice(fullName.lastIndexOf('.'));
          resolve({
            output: `http://127.0.0.1:1127/assets/result/${name}_output${ext}`,
          });
        },
      );

      workerProcess.on('exit', (code) => {
        console.log(`子进程已退出，退出码 ${code}`);
        // resolve(code);
      });
    });
  }
}
import { Injectable } from '@nestjs/common';

import * as child_process from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

import * as moment from 'moment';

@Injectable()
export class AppService {
  readonly basePath = path.join(__dirname, '../public/temp');

  getHello(): string {
    return 'Hello World!';
  }

  getPath(file): string {
    const date = moment().format('YYYY-MM-DD');
    return path.join(this.basePath,`./${date}/${file.originalname}`);
  }

  uploadFile(file, storagePath = '') {
    if (!storagePath) {
      storagePath = this.getPath(file);
    }
    console.debug(file);
    const writeStream = fs.createWriteStream(storagePath);
    writeStream.write(file.buffer);
    return new Promise((resolve) => {
      writeStream.on('drain', () => {
        console.debug('done');
        resolve(true);
      })
    })
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
          resolve(stdout);
        },
      );
  
      workerProcess.on('exit', (code) => {
        console.log(`子进程已退出，退出码 ${code}`);
        resolve(code);
      });
    })
  }
}
